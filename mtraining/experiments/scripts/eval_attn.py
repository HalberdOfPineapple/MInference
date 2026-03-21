#!/usr/bin/env python3

# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.distributed as dist
import yaml

from minference.configs.model2path import BASE_DIR as SPARSE_PATTERN_CONFIG_DIR
from mtraining.attn_funcs import AttnType
from mtraining.utils.cuda_timer import CudaEventTimer, get_cuda_timer, summarize_ms
from mtraining.attn_funcs.dense_func import (
    fa_attn_forward,
    wrap_striped_attn_func,
    wrap_zigzag_attn_func,
)
from mtraining.attn_funcs.minfer_func import (
    MINFER_IMPLEMENTATIONS,
    minfer_op,
)
from mtraining.attn_funcs.moba_func import (
    wrapped_moba_func,
    wrapped_moba_zigzag_func,
)
from mtraining.attn_funcs.xattn_func import (
    wrapped_xattn_func,
    XATTN_IMPLEMENTATIONS,
)


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
ATTN_TYPE_CHOICES = [
    AttnType.DENSE,
    AttnType.ZIGZAG_RING,
    AttnType.STRIPE_RING,
    AttnType.MINFER,
    AttnType.MOBA,
    AttnType.XATTN,
]


def parse_int_list(raw: str) -> List[int]:
    if raw is None or raw.strip().lower() in {"", "all"}:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def setup_distributed(timeout_minutes: int) -> Tuple[int, int, int]:
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=timeout_minutes),
        )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def load_yaml(path: str) -> Dict[str, Any]:
    if path is None or path.lower() == "none":
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def maybe_resolve_minfer_pattern_path(attn_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if "config_path" in attn_cfg:
        return attn_cfg
    if "pattern_config_name" in attn_cfg:
        out = dict(attn_cfg)
        out["config_path"] = os.path.join(
            SPARSE_PATTERN_CONFIG_DIR,
            f"{attn_cfg['pattern_config_name']}.json",
        )
        return out
    return attn_cfg


def gather_eval_pairs(
    qkv_root: str,
    world_size: int,
    layer_filter: List[int],
    sample_filter: List[int],
    max_pairs: int,
    rank: int,
) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    if rank == 0:
        root = Path(qkv_root)
        if not root.exists():
            raise FileNotFoundError(f"QKV root does not exist: {qkv_root}")

        layer_dirs = sorted(
            [p for p in root.glob("layer_*") if p.is_dir()],
            key=lambda p: int(p.name.split("_")[-1]),
        )
        if layer_filter:
            layer_dirs = [
                d for d in layer_dirs if int(d.name.split("_")[-1]) in set(layer_filter)
            ]

        sample_filter_set = set(sample_filter)
        for layer_dir in layer_dirs:
            layer_idx = int(layer_dir.name.split("_")[-1])
            sample_dirs = sorted(
                [p for p in layer_dir.glob("sample_*") if p.is_dir()],
                key=lambda p: int(p.name.split("_")[-1]),
            )
            for sample_dir in sample_dirs:
                sample_idx = int(sample_dir.name.split("_")[-1])
                if sample_filter_set and sample_idx not in sample_filter_set:
                    continue

                all_exist = all(
                    (sample_dir / f"qkv_{r}.pt").exists() for r in range(world_size)
                )
                if all_exist:
                    pairs.append((layer_idx, sample_idx))
                if max_pairs > 0 and len(pairs) >= max_pairs:
                    break
            if max_pairs > 0 and len(pairs) >= max_pairs:
                break

    obj = [pairs]
    dist.broadcast_object_list(obj, src=0)
    return obj[0]


def split_qkv_shard(
    qkv: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = qkv[:, :, :num_q_heads, :]
    k = qkv[:, :, num_q_heads : num_q_heads + num_kv_heads, :]
    v = qkv[:, :, num_q_heads + num_kv_heads :, :]
    return q, k, v


def infer_num_kv_heads(total_heads: int, num_q_heads: int, user_num_kv_heads: int) -> int:
    if user_num_kv_heads > 0:
        return user_num_kv_heads
    rem = total_heads - num_q_heads
    if rem <= 0 or rem % 2 != 0:
        raise ValueError(
            f"Cannot infer num_kv_heads from total_heads={total_heads}, num_q_heads={num_q_heads}"
        )
    return rem // 2


# summarize_ms is imported from mtraining.utils.cuda_timer


def build_runner(
    args: argparse.Namespace,
    layer_idx: int,
    q_bhnd: torch.Tensor,
    k_bhnd: torch.Tensor,
    v_bhnd: torch.Tensor,
    q_bnhd: torch.Tensor,
    k_bnhd: torch.Tensor,
    v_bnhd: torch.Tensor,
    ring_ranks: List[int],
    attn_cfg: Dict[str, Any],
) -> Callable[[], torch.Tensor]:
    # Minimal module stub used by attn wrappers.
    module = SimpleNamespace(
        layer_idx=layer_idx,
        head_dim=q_bhnd.shape[-1],
        config=SimpleNamespace(num_attention_heads=args.num_q_heads),
    )
    softmax_scale = (q_bhnd.shape[-1]) ** (-0.5)
    attn_type = args.attn_type

    if attn_type == AttnType.DENSE:
        return lambda: fa_attn_forward(
            module,
            q_bhnd,
            k_bhnd,
            v_bhnd,
            attention_mask=None,
            dropout=0.0,
            scaling=softmax_scale,
        )[0]

    if attn_type == AttnType.ZIGZAG_RING:
        return lambda: wrap_zigzag_attn_func(
            q_bnhd,
            k_bnhd,
            v_bnhd,
            layer_idx=layer_idx,
            softmax_scale=softmax_scale,
            dropout_p=0.0,
            causal=True,
            process_group=ring_ranks,
        )

    if attn_type == AttnType.STRIPE_RING:
        granularity = int(attn_cfg.get("granularity", args.granularity))
        return lambda: wrap_striped_attn_func(
            q_bnhd,
            k_bnhd,
            v_bnhd,
            layer_idx=layer_idx,
            granularity=granularity,
            softmax_scale=softmax_scale,
            dropout_p=0.0,
            causal=True,
            process_group=ring_ranks,
        )

    if attn_type == AttnType.MINFER:
        cfg = maybe_resolve_minfer_pattern_path(attn_cfg)
        implementation = cfg.get("implementation", "default")
        granularity = int(cfg.get("granularity", args.granularity))
        config_path = cfg.get("config_path")
        if config_path is None:
            raise ValueError(
                "minfer requires either `config_path` or `pattern_config_name` in train_attn_config_path"
            )
        with open(config_path, "r", encoding="utf-8") as f:
            all_pattern_dict = json.load(f)
        pattern_dict = {int(k): v for k, v in all_pattern_dict[layer_idx].items()}
        head_indices = torch.arange(
            args.num_q_heads, device=q_bhnd.device, dtype=torch.int32
        )

        impl_fn = MINFER_IMPLEMENTATIONS[implementation]
        if implementation == "default":
            return lambda: minfer_op(
                q_bhnd,
                k_bhnd,
                v_bhnd,
                head_indices,
                q_bhnd.shape[0],
                q_bhnd.shape[2],
                q_bhnd.shape[3],
                layer_idx,
                pattern_dict,
                0.0,
                granularity,
                None,
            )

        return lambda: impl_fn(
            q_bhnd,
            k_bhnd,
            v_bhnd,
            head_indices,
            q_bhnd.shape[0],
            q_bhnd.shape[2],
            q_bhnd.shape[3],
            layer_idx,
            pattern_dict,
            0.0,
            granularity,
            ring_ranks,
        )

    if attn_type == AttnType.MOBA:
        implementation = attn_cfg.get("implementation", "default")
        moba_topk = int(attn_cfg["moba_topk"])
        moba_chunk_size = int(attn_cfg["moba_chunk_size"])
        global_seq_len = int(args.global_seq_len)
        if global_seq_len <= 0:
            global_seq_len = int(q_bnhd.shape[1] * len(ring_ranks))

        if implementation == "default":
            return lambda: wrapped_moba_func(
                q_bnhd,
                k_bnhd,
                v_bnhd,
                global_seq_len,
                moba_topk,
                moba_chunk_size,
            )
        if implementation == "zigzag":
            return lambda: wrapped_moba_zigzag_func(
                q_bnhd,
                k_bnhd,
                v_bnhd,
                global_seq_len,
                moba_topk,
                moba_chunk_size,
                layer_idx,
                attention_mask=None,
                dropout=0.0,
                softmax_scale=softmax_scale,
                sliding_window=None,
                softcap=None,
                process_group=ring_ranks,
            )
        raise ValueError(f"Unsupported moba implementation: {implementation}")

    if attn_type == AttnType.XATTN:
        implementation = attn_cfg.get("implementation", "default")
        granularity = int(attn_cfg.get("granularity", args.granularity))
        xattn_params = dict(attn_cfg)
        xattn_params.pop("implementation", None)
        xattn_params.pop("granularity", None)
        head_indices = torch.arange(args.num_q_heads, device=q_bhnd.device)

        if implementation == "default":
            return lambda: wrapped_xattn_func(
                q_bnhd,
                k_bnhd,
                v_bnhd,
                head_indices,
                granularity,
                xattn_params,
                dropout=0.0,
                scaling=softmax_scale,
                sliding_window=None,
            )

        if implementation in XATTN_IMPLEMENTATIONS:
            print(f"{__name__} | Using xattn implementation: {implementation}")
            xattn_ring_fn = XATTN_IMPLEMENTATIONS[implementation]
            return lambda: xattn_ring_fn(
                q_bnhd,
                k_bnhd,
                v_bnhd,
                layer_idx,
                granularity,
                xattn_params,
                causal=True,
                dropout=0.0,
                scaling=softmax_scale,
                sliding_window=None,
                process_group=ring_ranks,
            )
        raise ValueError(f"Unsupported xattn implementation: {implementation}")

    raise ValueError(f"Unsupported attn_type: {attn_type}")


def benchmark_kernel(
    fn: Callable[[], torch.Tensor],
    warmup_iters: int,
    bench_iters: int,
    measure_backward: bool = False,
    timer: CudaEventTimer | None = None,
) -> Dict[str, Any]:
    """Benchmark attention kernel latency.

    All iterations are enqueued on the CUDA stream **without** intermediate
    CPU-GPU synchronization so that kernel-launch overhead is amortised
    across the loop.  Paired CUDA events recorded on the stream still give
    accurate *per-iteration* GPU-side timing.  A single
    ``torch.cuda.synchronize()`` after the loop ensures every event has
    completed before we read back elapsed times.

    Parameters
    ----------
    fn : callable
        Forward function returning the attention output tensor.
    warmup_iters : int
        Number of untimed warm-up iterations.
    bench_iters : int
        Number of timed iterations.
    measure_backward : bool
        If ``True``, also time ``output.backward(grad)`` after each forward.
        Input tensors must have ``requires_grad=True``.
    timer : CudaEventTimer or None
        When supplied, the timer is enabled during the bench loop so that
        fine-grained ``timer.region(...)`` calls inside the operator are
        recorded.
    """
    grad_out: torch.Tensor | None = None

    # ---- warm-up (untimed) -----------------------------------------------
    for _ in range(warmup_iters):
        out = fn()
        if measure_backward:
            if grad_out is None or grad_out.shape != out.shape:
                grad_out = torch.ones_like(out)
            out.backward(grad_out, retain_graph=False)
    torch.cuda.synchronize()

    # ---- timed iterations ------------------------------------------------
    # Record all event pairs on the stream without synchronising in between.
    # This lets the GPU pipeline iterations and amortises CPU-side kernel
    # launch overhead.  Event.elapsed_time() still returns the true GPU-side
    # duration between each (start, end) pair.
    if timer is not None:
        timer.reset()
        timer.enable()

    fwd_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
    bwd_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []

    for _ in range(bench_iters):
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)

        fwd_start.record()
        out = fn()
        fwd_end.record()

        if measure_backward:
            bwd_start = torch.cuda.Event(enable_timing=True)
            bwd_end = torch.cuda.Event(enable_timing=True)
            if grad_out is None or grad_out.shape != out.shape:
                grad_out = torch.ones_like(out)
            bwd_start.record()
            out.backward(grad_out, retain_graph=False)
            bwd_end.record()
            bwd_events.append((bwd_start, bwd_end))

        fwd_events.append((fwd_start, fwd_end))

    # Single sync after the full loop — all events are now complete.
    torch.cuda.synchronize()

    if timer is not None:
        timer.disable()

    # ---- collect results -------------------------------------------------
    fwd_ms = [s.elapsed_time(e) for s, e in fwd_events]
    result: Dict[str, Any] = {"fwd": summarize_ms(fwd_ms)}
    if measure_backward:
        bwd_ms = [s.elapsed_time(e) for s, e in bwd_events]
        result["bwd"] = summarize_ms(bwd_ms)
        result["total"] = summarize_ms(
            [f + b for f, b in zip(fwd_ms, bwd_ms)]
        )
    if timer is not None and timer.region_names:
        result["regions"] = timer.summarize()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate distributed attention latency from dumped QKV shards."
    )
    parser.add_argument("--qkv_dump_root", type=str, required=True)
    parser.add_argument(
        "--attn_type",
        type=str,
        default=AttnType.DENSE,
        choices=ATTN_TYPE_CHOICES,
    )
    parser.add_argument(
        "--train_attn_config_path",
        type=str,
        default=None,
        help="Optional YAML config used by minfer/moba/xattn",
    )
    parser.add_argument("--num_q_heads", type=int, default=16)
    parser.add_argument("--num_kv_heads", type=int, default=2)
    parser.add_argument("--global_seq_len", type=int, default=524288)
    parser.add_argument("--granularity", type=int, default=128)
    parser.add_argument("--layer_indices", type=str, default="all")
    parser.add_argument("--sample_indices", type=str, default="all")
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--warmup_iters", type=int, default=20)
    parser.add_argument("--bench_iters", type=int, default=100)
    parser.add_argument("--dtype", type=str, default="bf16", choices=DTYPE_MAP.keys())
    parser.add_argument("--timeout_minutes", type=int, default=30)
    parser.add_argument("--measure_backward", action="store_true",
                        help="Also measure backward pass latency. "
                             "Input tensors must support autograd.")
    parser.add_argument("--enable_region_timer", action="store_true",
                        help="Enable fine-grained CUDA event region timer. "
                             "Operators instrumented with get_cuda_timer().region() "
                             "will have their sub-regions recorded.")
    parser.add_argument("--save_json", type=str, default=None)
    parser.add_argument("--save_csv", type=str, default=None)
    args = parser.parse_args()

    rank, world_size, _ = setup_distributed(args.timeout_minutes)
    dtype = DTYPE_MAP[args.dtype]
    ring_ranks = list(range(world_size))
    attn_cfg = load_yaml(args.train_attn_config_path)
    timer = get_cuda_timer() if args.enable_region_timer else None

    if rank == 0:
        print("=" * 80)
        print("QKV Attention Latency Evaluation")
        print(f"attn_type={args.attn_type}")
        print(f"qkv_dump_root={args.qkv_dump_root}")
        print(f"world_size={world_size}")
        print(f"dtype={args.dtype}")
        print(f"warmup_iters={args.warmup_iters}, bench_iters={args.bench_iters}")
        print(f"measure_backward={args.measure_backward}")
        print(f"enable_region_timer={args.enable_region_timer}")
        print(f"train_attn_config={attn_cfg}")
        print("=" * 80, flush=True)

    layer_filter = parse_int_list(args.layer_indices)
    sample_filter = parse_int_list(args.sample_indices)


    # Gather all (layer_idx, sample_idx) pairs to evaluate
    pairs = gather_eval_pairs(
        qkv_root=args.qkv_dump_root,
        world_size=world_size,
        layer_filter=layer_filter,
        sample_filter=sample_filter,
        max_pairs=args.max_pairs,
        rank=rank,
    )
    if not pairs:
        raise RuntimeError("No valid (layer_idx, sample_idx) pair found under qkv_dump_root")

    all_rank_results: List[Dict[str, Any]] = []
    for layer_idx, sample_idx in pairs:
        qkv_path = (
            Path(args.qkv_dump_root)
            / f"layer_{layer_idx}"
            / f"sample_{sample_idx}"
            / f"qkv_{rank}.pt"
        )
        if not qkv_path.exists():
            raise FileNotFoundError(f"Missing QKV shard for rank {rank}: {qkv_path}")

        qkv = torch.load(qkv_path, map_location="cpu").to(device="cuda", dtype=dtype)
        num_kv_heads = infer_num_kv_heads(
            total_heads=qkv.shape[2],
            num_q_heads=args.num_q_heads,
            user_num_kv_heads=args.num_kv_heads,
        )
        q, k, v = split_qkv_shard(qkv, args.num_q_heads, num_kv_heads)
        q_bnhd, k_bnhd, v_bnhd = q.contiguous(), k.contiguous(), v.contiguous()
        q_bhnd = q_bnhd.permute(0, 2, 1, 3).contiguous()
        k_bhnd = k_bnhd.permute(0, 2, 1, 3).contiguous()
        v_bhnd = v_bnhd.permute(0, 2, 1, 3).contiguous()

        if args.measure_backward:
            q_bnhd.requires_grad_(True)
            k_bnhd.requires_grad_(True)
            v_bnhd.requires_grad_(True)
            q_bhnd.requires_grad_(True)
            k_bhnd.requires_grad_(True)
            v_bhnd.requires_grad_(True)

        runner = build_runner(
            args=args,
            layer_idx=layer_idx,
            q_bhnd=q_bhnd,
            k_bhnd=k_bhnd,
            v_bhnd=v_bhnd,
            q_bnhd=q_bnhd,
            k_bnhd=k_bnhd,
            v_bnhd=v_bnhd,
            ring_ranks=ring_ranks,
            attn_cfg=attn_cfg,
        )

        dist.barrier()
        stats = benchmark_kernel(
            fn=runner,
            warmup_iters=args.warmup_iters,
            bench_iters=args.bench_iters,
            measure_backward=args.measure_backward,
            timer=timer,
        )
        dist.barrier()

        local_result = {
            "rank": rank,
            "layer_idx": layer_idx,
            "sample_idx": sample_idx,
            "local_seq_len": int(q_bnhd.shape[1]),
            "num_q_heads": int(args.num_q_heads),
            "num_kv_heads": int(num_kv_heads),
            "head_dim": int(q_bnhd.shape[-1]),
            **stats["fwd"],          # backward compat: flat fwd stats
            "latency": stats,         # nested breakdown (fwd, bwd, total, regions)
        }
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_result)
        if rank == 0:
            all_rank_results.extend(gathered)
            by_rank = sorted(gathered, key=lambda x: x["rank"])
            if args.measure_backward:
                info = [
                    f"r{x['rank']}: fwd={x['latency']['fwd']['mean_ms']:.3f}ms"
                    f" bwd={x['latency']['bwd']['mean_ms']:.3f}ms"
                    for x in by_rank
                ]
            else:
                info = [f"r{x['rank']}={x['mean_ms']:.3f}ms" for x in by_rank]
            print(
                f"[layer={layer_idx} sample={sample_idx}] " + ", ".join(info),
                flush=True,
            )

    if rank == 0:
        summary = {
            "attn_type": args.attn_type,
            "world_size": world_size,
            "num_pairs": len(pairs),
            "num_rank_records": len(all_rank_results),
            "measure_backward": args.measure_backward,
        }
        for phase in ["fwd"] + (["bwd", "total"] if args.measure_backward else []):
            phase_means = [
                x["latency"][phase]["mean_ms"] for x in all_rank_results
            ]
            phase_summary = summarize_ms(phase_means)
            summary[f"{phase}_overall_mean_ms"] = phase_summary["mean_ms"]
            summary[f"{phase}_overall_p95_ms"] = phase_summary["p95_ms"]
        print("-" * 80)
        print(json.dumps(summary, indent=2))
        print("-" * 80, flush=True)

        if args.save_json:
            payload = {
                "args": vars(args),
                "summary": summary,
                "results": all_rank_results,
            }
            out = Path(args.save_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Saved JSON: {out}", flush=True)

        if args.save_csv:
            import csv

            def _flatten_dict(
                d: Dict[str, Any], prefix: str = "",
            ) -> Dict[str, Any]:
                flat: Dict[str, Any] = {}
                for k, v in d.items():
                    key = f"{prefix}{k}" if prefix else k
                    if isinstance(v, dict):
                        flat.update(_flatten_dict(v, f"{key}_"))
                    else:
                        flat[key] = v
                return flat

            flat_results = [_flatten_dict(row) for row in all_rank_results]
            all_keys: set = set()
            for flat in flat_results:
                all_keys.update(flat.keys())
            fieldnames = sorted(all_keys)

            out = Path(args.save_csv)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=fieldnames, extrasaction="ignore",
                )
                writer.writeheader()
                for row in flat_results:
                    writer.writerow(row)
            print(f"Saved CSV: {out}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
