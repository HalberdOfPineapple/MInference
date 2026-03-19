# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import torch
import torch.distributed as dist
from typing import Any, Dict, Optional, Tuple

from minference.ops.xattention_fa import xattn_estimate
from minference.ops.pit_sparse_flash_attention_v3 import block_attn_bwd, block_attn_fwd
from minference.dist_ops.utils import (
    RingComm,
    get_inner_ring,
    get_outer_ring,
    recover_striped_output,
    shuffle_block_mask_striped,
    shuffle_striped_input,
    update_out_and_lse,
)


def xattn_dr_stripe_forward_inner(
    process_group: dist.ProcessGroup,
    outer_step: int,
    outer_offset: int,
    inner_ring,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_mask: torch.Tensor,
    layer_idx: int,
    softmax_scale: float,
    granularity: int = 128,
    block_idx: Optional[torch.Tensor] = None,
    block_cnt: Optional[torch.Tensor] = None,
):
    inner_comm = RingComm(process_group, False, inner_ring)
    inner_rank = inner_ring.index(inner_comm.rank)
    num_inner_steps = len(inner_ring)
    next_k, next_v = None, None

    for inner_step in range(num_inner_steps):
        if inner_step + 1 != num_inner_steps:
            next_k, next_v = inner_comm.send_recv_kv(k, v)

        inner_offset = (inner_rank - inner_step) % num_inner_steps
        # Map the 2-D outer/inner traversal back to the equivalent flat ring step.
        offset = outer_offset * num_inner_steps + inner_offset
        block_mask_step = block_mask[inner_offset]
        block_causal = (outer_step == 0) and (inner_step == 0)

        block_out, block_lse = block_attn_fwd(
            q,
            k,
            v,
            block_mask=block_mask_step,
            softmax_scale=softmax_scale,
            granularity=granularity,
            causal=block_causal,
            step_idx=offset,
        )
        out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if inner_step + 1 != num_inner_steps:
            inner_comm.wait()
            k, v = next_k, next_v

    return out, lse


def xattn_dr_stripe_forward_outer(
    process_group: dist.ProcessGroup,
    outer_ring,
    inner_ring,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    layer_idx: int,
    softmax_scale: float,
    granularity: int = 128,
    block_idx: Optional[torch.Tensor] = None,
    block_cnt: Optional[torch.Tensor] = None,
):
    outer_comm = RingComm(process_group, False, outer_ring)
    outer_rank = outer_ring.index(outer_comm.rank)
    num_outer_steps = len(outer_ring)
    inner_block_masks = block_mask.chunk(num_outer_steps, dim=0)

    out, lse = None, None
    next_k, next_v = None, None

    for outer_step in range(num_outer_steps):
        if outer_step + 1 != num_outer_steps:
            next_k, next_v = outer_comm.send_recv_kv(k, v)

        outer_offset = (outer_rank - outer_step) % num_outer_steps
        out, lse = xattn_dr_stripe_forward_inner(
            process_group,
            outer_step,
            outer_offset,
            inner_ring,
            q,
            k,
            v,
            out,
            lse,
            inner_block_masks[outer_offset],
            layer_idx,
            softmax_scale,
            granularity=granularity,
            block_idx=None,
            block_cnt=None,
        )

        if outer_step + 1 != num_outer_steps:
            outer_comm.wait()
            k, v = next_k, next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


def xattn_dr_stripe_backward_inner(
    process_group: dist.ProcessGroup,
    outer_step: int,
    outer_offset: int,
    inner_ring,
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    layer_idx: int,
    softmax_scale: float,
    block_mask: torch.Tensor,
    granularity: int = 128,
    block_idx: Optional[torch.Tensor] = None,
    block_cnt: Optional[torch.Tensor] = None,
):
    inner_kv_comm = RingComm(process_group, False, inner_ring)
    inner_d_kv_comm = RingComm(process_group, False, inner_ring)
    inner_rank = inner_ring.index(inner_kv_comm.rank)
    num_inner_steps = len(inner_ring)

    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    for inner_step in range(num_inner_steps):
        if inner_step + 1 != num_inner_steps:
            next_k, next_v = inner_kv_comm.send_recv_kv(k, v)

        inner_offset = (inner_rank - inner_step) % num_inner_steps
        offset = outer_offset * num_inner_steps + inner_offset
        block_mask_step = block_mask[inner_offset]
        block_causal = (outer_step == 0) and (inner_step == 0)

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

        if inner_step == 0:
            dq = step_dq.to(torch.float32)
            dk = step_dk.to(torch.float32)
            dv = step_dv.to(torch.float32)
        else:
            inner_d_kv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv

            dq += step_dq
            dk += step_dk
            dv += step_dv

        if inner_step + 1 != num_inner_steps:
            inner_kv_comm.wait()
            k, v = next_k, next_v
        next_dk, next_dv = inner_d_kv_comm.send_recv_kv(
            dk, dv, dk_comm_buffer, dv_comm_buffer
        )

    inner_d_kv_comm.wait()
    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


def xattn_dr_stripe_backward_outer(
    process_group: dist.ProcessGroup,
    outer_ring,
    inner_ring,
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    layer_idx: int,
    softmax_scale: float,
    block_mask: torch.Tensor,
    granularity: int = 128,
    block_idx: Optional[torch.Tensor] = None,
    block_cnt: Optional[torch.Tensor] = None,
):
    outer_kv_comm = RingComm(process_group, False, outer_ring)
    outer_d_kv_comm = RingComm(process_group, False, outer_ring)
    outer_rank = outer_ring.index(outer_kv_comm.rank)
    num_outer_steps = len(outer_ring)
    inner_block_masks = block_mask.chunk(num_outer_steps, dim=0)

    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None
    next_k, next_v = None, None
    dk_comm_buffer, dv_comm_buffer = None, None

    for outer_step in range(num_outer_steps):
        if outer_step + 1 != num_outer_steps:
            next_k, next_v = outer_kv_comm.send_recv_kv(k, v)

        outer_offset = (outer_rank - outer_step) % num_outer_steps
        step_dq, step_dk, step_dv = xattn_dr_stripe_backward_inner(
            process_group,
            outer_step,
            outer_offset,
            inner_ring,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            layer_idx,
            softmax_scale,
            inner_block_masks[outer_offset],
            granularity=granularity,
            block_idx=None,
            block_cnt=None,
        )

        if outer_step == 0:
            dq = step_dq.to(torch.float32)
            dk = step_dk.to(torch.float32)
            dv = step_dv.to(torch.float32)
        else:
            outer_d_kv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv

            dq += step_dq
            dk += step_dk
            dv += step_dv

        if outer_step + 1 != num_outer_steps:
            outer_kv_comm.wait()
            k, v = next_k, next_v
        next_dk, next_dv = outer_d_kv_comm.send_recv_kv(
            dk, dv, dk_comm_buffer, dv_comm_buffer
        )

    outer_d_kv_comm.wait()
    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


class XAttnDRStripeFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx,
        xattn_params,
        granularity,
        causal,
        softmax_scale,
        return_softmax,
        deterministic,
        group,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        inner_ring = get_inner_ring(group)
        outer_ring = get_outer_ring(group)
        _, block_mask = xattn_estimate(
            q.transpose(1, 2),
            k.transpose(1, 2),
            block_size=granularity,
            ring_attn=True,
            **xattn_params,
        )

        q = shuffle_striped_input(
            to_send=q, dim=1, granularity=granularity, process_group=group
        )
        k = shuffle_striped_input(
            to_send=k, dim=1, granularity=granularity, process_group=group
        )
        v = shuffle_striped_input(
            to_send=v, dim=1, granularity=granularity, process_group=group
        )

        block_mask = shuffle_block_mask_striped(block_mask, group=group).to(q.device)
        block_mask = block_mask.contiguous()

        out, softmax_lse = xattn_dr_stripe_forward_outer(
            group,
            outer_ring,
            inner_ring,
            q,
            k,
            v,
            block_mask,
            layer_idx,
            softmax_scale,
            granularity=granularity,
            block_idx=None,
            block_cnt=None,
        )

        recovered_out = recover_striped_output(
            out, dim=1, granularity=granularity, process_group=group
        )
        if return_softmax:
            recovered_softmax_lse = recover_striped_output(
                softmax_lse, dim=2, granularity=granularity, process_group=group
            )

        ctx.save_for_backward(q, k, v, out, softmax_lse, block_mask)
        ctx.softmax_scale = softmax_scale
        ctx.granularity = granularity
        ctx.group = group
        ctx.inner_ring = inner_ring
        ctx.outer_ring = outer_ring
        ctx.layer_idx = layer_idx

        if return_softmax:
            return (recovered_out, recovered_softmax_lse, None)
        return recovered_out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, block_mask = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale
        granularity = ctx.granularity
        layer_idx = ctx.layer_idx
        group = ctx.group

        dout = shuffle_striped_input(
            to_send=dout, granularity=granularity, dim=1, process_group=group
        )

        dq, dk, dv = xattn_dr_stripe_backward_outer(
            group,
            ctx.outer_ring,
            ctx.inner_ring,
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
            block_idx=None,
            block_cnt=None,
        )

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


def xattn_dr_stripe_qkvpacked_func(
    qkv: torch.Tensor,
    layer_idx: int,
    xattn_params: Dict[str, Any],
    granularity: int = 128,
    dropout_p: int = 0.0,
    softmax_scale: float = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
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
    return XAttnDRStripeFunc.apply(
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


def xattn_dr_stripe_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    layer_idx: int,
    xattn_params: Dict[str, Any],
    granularity: int = 128,
    dropout_p: int = 0.0,
    softmax_scale: float = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
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

    return XAttnDRStripeFunc.apply(
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


def xattn_dr_stripe_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer_idx: int,
    xattn_params: Dict[str, Any],
    granularity: int = 128,
    dropout_p: int = 0.0,
    softmax_scale: float = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
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

    return XAttnDRStripeFunc.apply(
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
