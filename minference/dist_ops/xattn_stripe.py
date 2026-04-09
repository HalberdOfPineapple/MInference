# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import triton

from minference.dist_ops.utils import (
    RingComm,
    recover_striped_output,
    shuffle_block_mask_striped,
    shuffle_striped_input,
    update_out_and_lse,
)
from minference.ops.pit_sparse_flash_attention_v3 import block_attn_bwd, block_attn_fwd
from minference.ops.xattention_fa import xattn_estimate


def xattn_stripe_forward(
    process_group: dist.ProcessGroup,
    q: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    block_mask: torch.Tensor,  # [world_size, batch_size, num_qo_heads, num_blocks, num_blocks]
    layer_idx: int,
    softmax_scale: float,
    granularity: int = 128,
    block_idx: Optional[torch.Tensor] = None,
    block_cnt: Optional[torch.Tensor] = None,
):
    comm = RingComm(process_group)
    out, lse = None, None
    next_k, next_v = None, None

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)

        # [batch_size, num_qo_heads, num_blocks_local, num_blocks_local]
        block_mask_step = block_mask[step]
        block_causal = step == 0

        block_out, block_lse = block_attn_fwd(
            q,
            k,
            v,
            block_mask=block_mask_step,
            softmax_scale=softmax_scale,
            granularity=granularity,
            causal=block_causal,
            step_idx=step,
        )

        out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v
    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


def xattn_stripe_backward(
    process_group: dist.ProcessGroup,
    dout: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    q: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_kv_heads, head_dim]
    out: torch.Tensor,  # [batch_size, num_tokens, num_qo_heads, head_dim]
    softmax_lse: torch.Tensor,  # [batch_size, num_qo_heads, num_tokens]
    layer_idx: int,
    softmax_scale: float,
    block_mask: torch.Tensor,  # [world_size, batch_size, num_qo_heads, num_blocks, num_blocks]
    granularity: int = 128,
    block_idx: Optional[
        torch.Tensor
    ] = None,  # [world_size, batch_size, num_qo_heads, num_blocks_local, num_blocks]
    block_cnt: Optional[
        torch.Tensor
    ] = None,  # [world_size, batch_size, num_qo_heads, num_blocks_local]
):
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)

    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)

        block_causal = step == 0
        block_mask_step = block_mask[step]

        # --------------------------------
        # Block Mask
        step_dq, step_dk, step_dv = block_attn_bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale,
            block_mask_step,
            granularity=granularity,
            deterministic=False,
            causal=block_causal,
        )

        # Update dQ, dK, dV
        if step == 0:
            # TODO: check if float32 is necessary
            dq = step_dq.to(torch.float32)
            dk = step_dk.to(torch.float32)
            dv = step_dv.to(torch.float32)
        else:
            d_kv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv

            dq += step_dq
            dk += step_dk
            dv += step_dv

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v
        next_dk, next_dv = d_kv_comm.send_recv_kv(
            dk, dv, dk_comm_buffer, dv_comm_buffer
        )

    d_kv_comm.wait()
    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


class XAttnStripeFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx,
        xattn_params,  # Dict[str, Any]
        granularity,
        causal,
        softmax_scale,
        return_softmax,
        deterministic,
        group,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        _, block_mask = xattn_estimate(
            q.transpose(1, 2),
            k.transpose(1, 2),
            block_size=granularity,
            ring_attn=True,
            **xattn_params
        )

        # ------------------------------------------------------------------
        # QKV Shuffling
        q = shuffle_striped_input(
            to_send=q, dim=1, granularity=granularity, process_group=group
        )
        k = shuffle_striped_input(
            to_send=k, dim=1, granularity=granularity, process_group=group
        )
        v = shuffle_striped_input(
            to_send=v, dim=1, granularity=granularity, process_group=group
        )

        # ------------------------------------------------------------------
        # Index Shuffling
        block_mask = shuffle_block_mask_striped(block_mask, group=group).to(q.device)

        # if use_triton():
        #     block_idx, block_cnt = convert_blockmask(block_mask, block_size_M=granularity, block_size_N=64)
        block_idx, block_cnt = None, None
        block_mask = block_mask.contiguous()

        # ----------------------------------------------
        # Compute
        out, softmax_lse = xattn_stripe_forward(
            group,
            q,
            k,
            v,
            block_mask,
            layer_idx,
            softmax_scale,
            granularity=granularity,
            block_idx=block_idx,
            block_cnt=block_cnt,
        )

        # ----------------------------------------------
        # Recover outputs
        recovered_out = recover_striped_output(
            out, dim=1, granularity=granularity, process_group=group
        )
        if return_softmax:
            recovered_softmax_lse = recover_striped_output(
                softmax_lse, dim=2, granularity=granularity, process_group=group
            )

        # -------------------------------
        # Variale Saving
        ctx.save_for_backward(q, k, v, out, softmax_lse, block_mask)
        ctx.softmax_scale = softmax_scale
        ctx.granularity = granularity
        ctx.group = group
        ctx.layer_idx = layer_idx

        # -------------------------------
        # Recover outputs
        if return_softmax:
            return (recovered_out, recovered_softmax_lse, None)
        return recovered_out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, block_mask = ctx.saved_tensors
        block_idx, block_cnt = None, None
        softmax_scale = ctx.softmax_scale
        granularity = ctx.granularity
        layer_idx = ctx.layer_idx
        group = ctx.group

        dout = shuffle_striped_input(
            to_send=dout, granularity=granularity, dim=1, process_group=group
        )

        # ----------------------------------------------
        # Compute
        dq, dk, dv = xattn_stripe_backward(
            group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            layer_idx,
            softmax_scale,
            block_mask,
            granularity,
            block_idx=block_idx,
            block_cnt=block_cnt,
        )

        # ----------------------------------------------
        # Recover
        dq = recover_striped_output(
            dq, granularity=granularity, dim=1, process_group=group
        )
        dk = recover_striped_output(
            dk, granularity=granularity, dim=1, process_group=group
        )
        dv = recover_striped_output(
            dv, granularity=granularity, dim=1, process_group=group
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def xattn_stripe_qkvpacked_func(
    qkv: torch.Tensor,  # [batch_size, num_tokens, 3, num_heads, head_dim]
    layer_idx: int,
    xattn_params: Dict[str, Any],
    granularity: int = 128,
    dropout_p: int = 0.0,
    softmax_scale: float = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    alibi_slopes: Tuple[float, float] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    group: dist.ProcessGroup = None,
):
    assert causal
    assert dropout_p == 0
    assert window_size == (-1, -1)
    assert alibi_slopes is None
    assert not deterministic
    return XAttnStripeFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        layer_idx,
        xattn_params,
        granularity,
        causal,
        softmax_scale,
        return_attn_probs,
        deterministic,
        group,
    )


def xattn_stripe_kvpacked_func(
    q: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    kv: torch.Tensor,  # [batch_size, num_tokens, 2, num_heads, head_dim]
    layer_idx: int,
    xattn_params: Dict[str, Any],
    granularity: int = 128,
    dropout_p: int = 0.0,
    softmax_scale: float = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    alibi_slopes: Tuple[float, float] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    group: dist.ProcessGroup = None,
):
    assert causal
    assert dropout_p == 0
    assert window_size == (-1, -1)
    assert alibi_slopes is None
    assert not deterministic

    return XAttnStripeFunc.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        layer_idx,
        xattn_params,
        granularity,
        causal,
        softmax_scale,
        return_attn_probs,
        deterministic,
        group,
    )


def xattn_stripe_func(  # the one used for nnscaler training
    q: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    k: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    v: torch.Tensor,  # [batch_size, num_tokens, num_heads, head_dim]
    layer_idx: int,
    xattn_params: Dict[str, Any],
    granularity: int = 128,
    dropout_p: int = 0.0,
    softmax_scale: float = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    alibi_slopes: Tuple[float, float] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    group: dist.ProcessGroup = None,
) -> torch.Tensor:
    assert causal
    assert dropout_p == 0
    assert window_size == (-1, -1)
    assert alibi_slopes is None
    assert not deterministic

    return XAttnStripeFunc.apply(
        q,
        k,
        v,
        layer_idx,
        xattn_params,
        granularity,
        causal,
        softmax_scale,
        return_attn_probs,
        deterministic,
        group,
    )
