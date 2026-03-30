#!/usr/bin/env python3
# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Unified analysis of sparse attention data from the inference collection pipeline.

Subcommands
-----------
ratio    — Sparse ratio trends across checkpoints / layers.
indices  — Vertical & slash index statistics (counts, fragmentation, overlap).
masks    — Block-mask density and bar-count distributions.

Data layout (produced by ``infer_sparse_indices.py`` / ``watch_and_infer.sh``)::

    <base_dir>/
    ├── 0000-0001/
    │   ├── sparse_ratios.json
    │   ├── indices/sample_0000.pt
    │   └── masks/sample_0000.pt
    ├── 0000-0002/
    │   └── ...

Usage::

    python analyze.py ratio   --base_dir <path> --mode summary
    python analyze.py ratio   --base_dir <path> --mode heatmap --output_dir ./plots
    python analyze.py indices --base_dir <path> --ckpt_tag 0000-0005 --mode summary
    python analyze.py indices --base_dir <path> --mode frag_over_steps
    python analyze.py masks   --base_dir <path> --ckpt_tag 0000-0005 --mode summary
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════

def discover_checkpoints(base_dir: str) -> List[str]:
    """Return sorted ``XXXX-XXXX`` checkpoint tags found under *base_dir*."""
    tags = []
    if not os.path.isdir(base_dir):
        return tags
    for name in os.listdir(base_dir):
        if re.match(r"^\d{4}-\d{4}$", name) and os.path.isdir(os.path.join(base_dir, name)):
            tags.append(name)
    return sorted(tags)


def ckpt_tag_to_step(tag: str) -> int:
    """``'0000-0039'`` → ``39`` (epoch × 10000 + iter)."""
    e, i = tag.split("-")
    return int(e) * 10000 + int(i)


def select_items(all_items: List[int], spec: Optional[str]) -> List[int]:
    """Parse a selection spec: ``'all'``, ``'auto'``, ``'0,5,10'``, ``'0-35:5'``."""
    if spec is None or spec == "auto":
        n = len(all_items)
        if n <= 12:
            return list(all_items)
        step = max(1, n // 12)
        return all_items[::step]
    if spec == "all":
        return list(all_items)
    if ":" in spec:
        m = re.match(r"(\d+)-(\d+):(\d+)", spec)
        if m:
            lo, hi, st = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return [x for x in all_items if lo <= x <= hi and (x - lo) % st == 0]
    return [int(x) for x in spec.split(",") if int(x) in all_items]


def select_ckpts(all_tags: List[str], spec: Optional[str]) -> List[str]:
    """Parse checkpoint selection: ``'all'``, ``'auto'``, ``'0000-0001,0000-0005'``."""
    if spec is None or spec == "auto":
        n = len(all_tags)
        return all_tags if n <= 10 else all_tags[:: max(1, n // 10)]
    if spec == "all":
        return list(all_tags)
    return [t.strip() for t in spec.split(",") if t.strip() in all_tags]


def save_or_show(fig, output_dir: Optional[str], default_name: str):
    """Save figure to *output_dir/default_name.png* or show interactively."""
    import matplotlib.pyplot as plt
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{default_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    else:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  RATIO subcommand
# ═══════════════════════════════════════════════════════════════════════════

def _load_ratios(base_dir: str, tags: List[str]) -> Dict[str, Dict[int, Dict[int, float]]]:
    """Load sparse ratio JSONs across checkpoints.

    Returns ``{tag: {sample_idx: {layer_idx: ratio}}}``.
    """
    data: Dict[str, Dict[int, Dict[int, float]]] = {}
    for tag in tags:
        path = os.path.join(base_dir, tag, "sparse_ratios.json")
        if not os.path.isfile(path):
            print(f"[warn] missing {path}", file=sys.stderr)
            continue
        with open(path) as f:
            raw = json.load(f)
        data[tag] = {int(s): {int(l): v for l, v in ldict.items()} for s, ldict in raw.items()}
    return data


def _ratios_to_numpy(data):
    """→ (array[ckpt, sample, layer], tags, samples, layers).  NaN for missing."""
    tags = sorted(data.keys())
    all_s = sorted({s for d in data.values() for s in d})
    all_l = sorted({l for d in data.values() for sd in d.values() for l in sd})
    arr = np.full((len(tags), len(all_s), len(all_l)), np.nan)
    si_map = {s: i for i, s in enumerate(all_s)}
    li_map = {l: i for i, l in enumerate(all_l)}
    for ci, t in enumerate(tags):
        for s, ldict in data[t].items():
            for l, v in ldict.items():
                arr[ci, si_map[s], li_map[l]] = v
    return arr, tags, all_s, all_l


# ---- ratio modes ----------------------------------------------------------

def ratio_summary(arr, tags, samples, layers, args):
    print(f"Checkpoints : {len(tags)} ({tags[0]} .. {tags[-1]})")
    print(f"Samples/ckpt: {len(samples)}")
    print(f"Layers      : {len(layers)}\n")
    print(f"{'Checkpoint':<14} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 50)
    for i, t in enumerate(tags):
        v = arr[i]
        print(f"{t:<14} {np.nanmean(v):8.4f} {np.nanstd(v):8.4f} "
              f"{np.nanmin(v):8.4f} {np.nanmax(v):8.4f}")
    print("-" * 50)
    print(f"{'Overall':<14} {np.nanmean(arr):8.4f} {np.nanstd(arr):8.4f} "
          f"{np.nanmin(arr):8.4f} {np.nanmax(arr):8.4f}")
    print(f"\n{'Layer':<8} {'Mean':>8} {'Std':>8}")
    print("-" * 28)
    for j, l in enumerate(layers):
        v = arr[:, :, j]
        print(f"{l:<8} {np.nanmean(v):8.4f} {np.nanstd(v):8.4f}")


def ratio_layer_over_steps(arr, tags, samples, layers, args):
    """Line plot: ratio vs step, one line per layer."""
    import matplotlib.pyplot as plt
    steps = [ckpt_tag_to_step(t) for t in tags]
    means = np.nanmean(arr, axis=1)
    show = select_items(layers, args.layers)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(show)))
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, li in enumerate(show):
        j = layers.index(li)
        ax.plot(steps, means[:, j], label=f"L{li}", color=cmap[idx], alpha=0.8)
    ax.set_xlabel("Training Step"); ax.set_ylabel("Sparse Ratio")
    ax.set_title("Sparse Ratio vs Training Step (per layer)")
    ax.legend(fontsize="small", ncol=max(1, len(show) // 12), loc="best")
    ax.grid(True, alpha=0.3)
    save_or_show(fig, args.output_dir, "ratio_layer_over_steps")


def ratio_step_over_layers(arr, tags, samples, layers, args):
    """Line plot: ratio vs layer, one line per checkpoint."""
    import matplotlib.pyplot as plt
    means = np.nanmean(arr, axis=1)
    show = select_ckpts(tags, args.ckpts)
    cmap = plt.cm.coolwarm(np.linspace(0, 1, len(show)))
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, t in enumerate(show):
        i = tags.index(t)
        ax.plot(layers, means[i], label=t, color=cmap[idx], alpha=0.8)
    ax.set_xlabel("Layer"); ax.set_ylabel("Sparse Ratio")
    ax.set_title("Sparse Ratio vs Layer (per checkpoint)")
    ax.legend(fontsize="small", ncol=max(1, len(show) // 8), loc="best")
    ax.grid(True, alpha=0.3)
    save_or_show(fig, args.output_dir, "ratio_step_over_layers")


def ratio_heatmap(arr, tags, samples, layers, args):
    """Heatmap: layers (x) × checkpoints (y)."""
    import matplotlib.pyplot as plt
    means = np.nanmean(arr, axis=1)
    fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.35), max(6, len(tags) * 0.25)))
    im = ax.imshow(means, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(layers))); ax.set_xticklabels(layers, fontsize=7)
    ax.set_yticks(range(len(tags))); ax.set_yticklabels(tags, fontsize=7)
    ax.set_xlabel("Layer"); ax.set_ylabel("Checkpoint")
    ax.set_title("Sparse Ratio Heatmap")
    fig.colorbar(im, ax=ax, label="Sparse Ratio"); fig.tight_layout()
    save_or_show(fig, args.output_dir, "ratio_heatmap")


def ratio_delta_heatmap(arr, tags, samples, layers, args):
    """Heatmap of ratio *change* from first checkpoint."""
    import matplotlib.pyplot as plt
    means = np.nanmean(arr, axis=1)
    delta = means - means[0:1, :]
    vabs = max(abs(np.nanmin(delta)), abs(np.nanmax(delta)), 0.01)
    fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.35), max(6, len(tags) * 0.25)))
    im = ax.imshow(delta, aspect="auto", cmap="RdBu_r", vmin=-vabs, vmax=vabs)
    ax.set_xticks(range(len(layers))); ax.set_xticklabels(layers, fontsize=7)
    ax.set_yticks(range(len(tags))); ax.set_yticklabels(tags, fontsize=7)
    ax.set_xlabel("Layer"); ax.set_ylabel("Checkpoint")
    ax.set_title(f"Sparse Ratio Δ from {tags[0]}")
    fig.colorbar(im, ax=ax, label="Δ Ratio"); fig.tight_layout()
    save_or_show(fig, args.output_dir, "ratio_delta_heatmap")


def ratio_global_trend(arr, tags, samples, layers, args):
    """Global mean ratio ± std over steps."""
    import matplotlib.pyplot as plt
    steps = [ckpt_tag_to_step(t) for t in tags]
    mu = np.nanmean(arr, axis=(1, 2))
    sd = np.nanstd(arr, axis=(1, 2))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, mu, color="steelblue", lw=2, label="Mean")
    ax.fill_between(steps, mu - sd, mu + sd, color="steelblue", alpha=0.2, label="±1σ")
    ax.set_xlabel("Training Step"); ax.set_ylabel("Sparse Ratio")
    ax.set_title("Global Sparse Ratio Trend"); ax.legend(); ax.grid(True, alpha=0.3)
    save_or_show(fig, args.output_dir, "ratio_global_trend")


def ratio_layer_avg(arr, tags, samples, layers, args):
    """Export per-layer mean/std (averaged across all checkpoints & samples) to CSV."""
    # arr shape: (ckpt, sample, layer) → flatten ckpt+sample → per-layer stats
    out = os.path.join(args.output_dir or ".", "sparse_ratio_layer_avg.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    global_mean = float(np.nanmean(arr))
    global_std = float(np.nanstd(arr))
    with open(out, "w") as f:
        f.write("layer,mean,std,min,max,global_mean,global_std\n")
        for li, l in enumerate(layers):
            vals = arr[:, :, li].flatten()
            vals = vals[~np.isnan(vals)]
            f.write(f"{l},{np.mean(vals):.6f},{np.std(vals):.6f},"
                    f"{np.min(vals):.6f},{np.max(vals):.6f},"
                    f"{global_mean:.6f},{global_std:.6f}\n")
    print(f"Exported → {out}")


def ratio_export_csv(arr, tags, samples, layers, args):
    out = os.path.join(args.output_dir or ".", "sparse_ratios.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("checkpoint,step,sample,layer,sparse_ratio\n")
        for ci, t in enumerate(tags):
            step = ckpt_tag_to_step(t)
            for si, s in enumerate(samples):
                for li, l in enumerate(layers):
                    v = arr[ci, si, li]
                    if not np.isnan(v):
                        f.write(f"{t},{step},{s},{l},{v:.6f}\n")
    print(f"Exported → {out}")


def ratio_draw_all(arr, tags, samples, layers, args):
    for name, fn in [("summary", ratio_summary),
                     ("global_trend", ratio_global_trend),
                     ("heatmap", ratio_heatmap),
                     ("delta_heatmap", ratio_delta_heatmap),
                     ("layer_over_steps", ratio_layer_over_steps),
                     ("step_over_layers", ratio_step_over_layers)]:
        print(f"  → {name}")
        fn(arr, tags, samples, layers, args)


RATIO_MODES = {
    "summary": ratio_summary,
    "layer_avg": ratio_layer_avg,
    "layer_over_steps": ratio_layer_over_steps,
    "step_over_layers": ratio_step_over_layers,
    "heatmap": ratio_heatmap,
    "delta_heatmap": ratio_delta_heatmap,
    "global_trend": ratio_global_trend,
    "export_csv": ratio_export_csv,
    "draw_all": ratio_draw_all,
}


def run_ratio(args):
    tags = discover_checkpoints(args.base_dir)
    if not tags:
        sys.exit(f"No checkpoints in {args.base_dir}")
    data = _load_ratios(args.base_dir, tags)
    if not data:
        sys.exit("No sparse_ratios.json files found.")
    arr, tags, samples, layers = _ratios_to_numpy(data)
    print(f"Loaded: {arr.shape[0]} ckpts, {arr.shape[1]} samples, {arr.shape[2]} layers\n")
    RATIO_MODES[args.mode](arr, tags, samples, layers, args)


# ═══════════════════════════════════════════════════════════════════════════
#  INDICES subcommand
# ═══════════════════════════════════════════════════════════════════════════

V_PAD = 2147483647  # INT32_MAX
S_PAD = -1

def _find_indices_dir(data_dir: str) -> str:
    sub = os.path.join(data_dir, "indices")
    return sub if os.path.isdir(sub) else data_dir

def _discover_index_samples(data_dir: str) -> List[int]:
    d = _find_indices_dir(data_dir)
    return sorted(int(m.group(1)) for name in os.listdir(d) if (m := re.match(r"sample_(\d+)\.pt$", name)))

def _load_index_sample(data_dir: str, si: int):
    import torch
    d = _find_indices_dir(data_dir)
    return torch.load(os.path.join(d, f"sample_{si:04d}.pt"), map_location="cpu")

def _load_all_indices(data_dir: str, sample_indices=None):
    if sample_indices is None:
        sample_indices = _discover_index_samples(data_dir)
    if not sample_indices:
        raise FileNotFoundError(f"No index samples in {data_dir}")
    samples_data = [_load_index_sample(data_dir, si) for si in sample_indices]
    first = samples_data[0]
    num_layers = len(first)
    num_heads = first[0]["v_idx"].shape[0]
    return samples_data, sample_indices, num_layers, num_heads

def _count_valid(t, pad):
    return (t != pad).sum(dim=-1)

def _get_valid_np(arr1d, pad):
    v = arr1d[arr1d != pad]
    return np.sort(v)

def _gap_stats(sv):
    n = len(sv)
    if n <= 1:
        return {"count": n, "frag": 0.0, "mean_run": float(n), "gap_mean": 0.0}
    gaps = np.diff(sv)
    breaks = np.where(gaps > 1)[0]
    nruns = len(breaks) + 1
    return {
        "count": n,
        "frag": (nruns - 1) / max(n - 1, 1),
        "mean_run": n / nruns,
        "gap_mean": float(np.mean(gaps)),
    }


# ---- indices modes (single-checkpoint) ------------------------------------

def idx_summary(data, si, nl, nh, args):
    """Print per-layer valid-index statistics."""
    v_c = np.zeros((len(data), nl, nh))
    s_c = np.zeros((len(data), nl, nh))
    for i, d in enumerate(data):
        for li in range(nl):
            v_c[i, li] = _count_valid(d[li]["v_idx"], V_PAD).numpy()
            s_c[i, li] = _count_valid(d[li]["s_idx"], S_PAD).numpy()
    va = data[0][0]["v_idx"].shape[-1]
    sa = data[0][0]["s_idx"].shape[-1]
    print(f"Samples={len(data)}  Layers={nl}  Heads={nh}")
    print(f"V alloc={va}  S alloc={sa}\n")
    print(f"{'Layer':<8} {'V mean':>8} {'V std':>8} {'S mean':>8} {'S std':>8}")
    print("-" * 44)
    for li in range(nl):
        print(f"{li:<8} {v_c[:, li].mean():8.1f} {v_c[:, li].std():8.1f} "
              f"{s_c[:, li].mean():8.1f} {s_c[:, li].std():8.1f}")


def idx_count_heatmap(data, si, nl, nh, args):
    """Heatmap: mean valid count per (layer, head)."""
    import matplotlib.pyplot as plt
    v_c = np.zeros((len(data), nl, nh))
    for i, d in enumerate(data):
        for li in range(nl):
            v_c[i, li] = _count_valid(d[li]["v_idx"], V_PAD).numpy()
    mean = v_c.mean(axis=0)
    fig, ax = plt.subplots(figsize=(max(8, nh * 0.4), max(6, nl * 0.25)))
    im = ax.imshow(mean, aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("Head"); ax.set_ylabel("Layer")
    ax.set_title("Mean Valid V-Index Count (layer × head)")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    save_or_show(fig, args.output_dir, "idx_count_heatmap")


def idx_fragmentation(data, si, nl, nh, args):
    """Fragmentation index per layer (mean over samples × heads)."""
    import matplotlib.pyplot as plt
    frags = np.zeros((len(data), nl, nh))
    for i, d in enumerate(data):
        for li in range(nl):
            for hi in range(nh):
                sv = _get_valid_np(d[li]["v_idx"][hi].numpy(), V_PAD)
                frags[i, li, hi] = _gap_stats(sv)["frag"]
    layer_frag = frags.mean(axis=(0, 2))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(nl), layer_frag, color="coral", alpha=0.8)
    ax.set_xlabel("Layer"); ax.set_ylabel("Fragmentation Index")
    ax.set_title("V-Index Fragmentation per Layer"); ax.grid(True, alpha=0.3, axis="y")
    save_or_show(fig, args.output_dir, "idx_fragmentation")


def idx_discreteness(data, si, nl, nh, args):
    """Print per-layer gap/run/fragmentation table."""
    print(f"{'Layer':<6} {'Count':>7} {'Frag':>7} {'MeanRun':>8} {'GapMean':>8}")
    print("-" * 40)
    for li in range(nl):
        stats_list = []
        for d in data:
            for hi in range(nh):
                sv = _get_valid_np(d[li]["v_idx"][hi].numpy(), V_PAD)
                stats_list.append(_gap_stats(sv))
        mc = np.mean([s["count"] for s in stats_list])
        mf = np.mean([s["frag"] for s in stats_list])
        mr = np.mean([s["mean_run"] for s in stats_list])
        mg = np.mean([s["gap_mean"] for s in stats_list])
        print(f"{li:<6} {mc:7.0f} {mf:7.3f} {mr:8.1f} {mg:8.1f}")


def idx_position_dist(data, si, nl, nh, args):
    """Histogram of V-index positions across all heads, selected layers."""
    import matplotlib.pyplot as plt
    show_layers = select_items(list(range(nl)), args.layers)
    fig, axes = plt.subplots(len(show_layers), 1, figsize=(12, 3 * len(show_layers)),
                             squeeze=False, sharex=True)
    for row, li in enumerate(show_layers):
        all_pos = []
        for d in data:
            v = d[li]["v_idx"].numpy().flatten()
            all_pos.append(v[v != V_PAD])
        all_pos = np.concatenate(all_pos)
        ax = axes[row][0]
        ax.hist(all_pos, bins=100, color="steelblue", alpha=0.7, edgecolor="none")
        ax.set_ylabel(f"L{li}")
    axes[-1][0].set_xlabel("Token Position")
    fig.suptitle("V-Index Position Distribution", fontsize=13)
    fig.tight_layout()
    save_or_show(fig, args.output_dir, "idx_position_dist")


def idx_export_csv(data, si, nl, nh, args):
    out = os.path.join(args.output_dir or ".", "indices.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write("sample,layer,head,v_valid,s_valid\n")
        for i, d in enumerate(data):
            for li in range(nl):
                for hi in range(nh):
                    vc = int((d[li]["v_idx"][hi] != V_PAD).sum())
                    sc = int((d[li]["s_idx"][hi] != S_PAD).sum())
                    f.write(f"{si[i]},{li},{hi},{vc},{sc}\n")
    print(f"Exported → {out}")


def idx_draw_all(data, si, nl, nh, args):
    for name, fn in [("summary", idx_summary), ("count_heatmap", idx_count_heatmap),
                     ("fragmentation", idx_fragmentation), ("discreteness", idx_discreteness),
                     ("position_dist", idx_position_dist)]:
        print(f"  → {name}")
        fn(data, si, nl, nh, args)


# ---- indices modes (cross-checkpoint) -------------------------------------

def idx_frag_over_steps(data, si, nl, nh, args):
    """Fragmentation vs training step, per layer."""
    import matplotlib.pyplot as plt
    tags = discover_checkpoints(args.base_dir)
    show_layers = select_items(list(range(nl)), args.layers)
    steps = [ckpt_tag_to_step(t) for t in tags]
    frag_per_step = np.zeros((len(tags), nl))
    for ti, tag in enumerate(tags):
        dd = os.path.join(args.base_dir, tag)
        sdata, _, _, _ = _load_all_indices(dd)
        for li in range(nl):
            fs = []
            for d in sdata:
                for hi in range(nh):
                    sv = _get_valid_np(d[li]["v_idx"][hi].numpy(), V_PAD)
                    fs.append(_gap_stats(sv)["frag"])
            frag_per_step[ti, li] = np.mean(fs)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(show_layers)))
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, li in enumerate(show_layers):
        ax.plot(steps, frag_per_step[:, li], label=f"L{li}", color=cmap[idx])
    ax.set_xlabel("Training Step"); ax.set_ylabel("Fragmentation")
    ax.set_title("V-Index Fragmentation vs Step")
    ax.legend(fontsize="small", ncol=max(1, len(show_layers) // 12)); ax.grid(True, alpha=0.3)
    save_or_show(fig, args.output_dir, "idx_frag_over_steps")


def idx_count_over_steps(data, si, nl, nh, args):
    """Valid index count vs training step, per layer."""
    import matplotlib.pyplot as plt
    tags = discover_checkpoints(args.base_dir)
    show_layers = select_items(list(range(nl)), args.layers)
    steps = [ckpt_tag_to_step(t) for t in tags]
    cnt = np.zeros((len(tags), nl))
    for ti, tag in enumerate(tags):
        dd = os.path.join(args.base_dir, tag)
        sdata, _, _, _ = _load_all_indices(dd)
        for li in range(nl):
            cs = []
            for d in sdata:
                cs.append(_count_valid(d[li]["v_idx"], V_PAD).float().mean().item())
            cnt[ti, li] = np.mean(cs)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(show_layers)))
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, li in enumerate(show_layers):
        ax.plot(steps, cnt[:, li], label=f"L{li}", color=cmap[idx])
    ax.set_xlabel("Training Step"); ax.set_ylabel("Mean Valid V-Count")
    ax.set_title("V-Index Count vs Step")
    ax.legend(fontsize="small", ncol=max(1, len(show_layers) // 12)); ax.grid(True, alpha=0.3)
    save_or_show(fig, args.output_dir, "idx_count_over_steps")


IDX_MODES = {
    "summary": idx_summary,
    "count_heatmap": idx_count_heatmap,
    "fragmentation": idx_fragmentation,
    "discreteness": idx_discreteness,
    "position_dist": idx_position_dist,
    "export_csv": idx_export_csv,
    "draw_all": idx_draw_all,
    "frag_over_steps": idx_frag_over_steps,
    "count_over_steps": idx_count_over_steps,
}

IDX_CROSS_MODES = {"frag_over_steps", "count_over_steps"}


def run_indices(args):
    tag = args.ckpt_tag
    tags = discover_checkpoints(args.base_dir)
    if not tags:
        sys.exit(f"No checkpoints in {args.base_dir}")
    if tag is None:
        tag = tags[-1]
    data_dir = os.path.join(args.base_dir, tag)
    print(f"Using checkpoint: {tag}")
    data, si, nl, nh = _load_all_indices(data_dir)
    print(f"Loaded {len(data)} samples, {nl} layers, {nh} heads\n")
    IDX_MODES[args.mode](data, si, nl, nh, args)


# ═══════════════════════════════════════════════════════════════════════════
#  MASKS subcommand
# ═══════════════════════════════════════════════════════════════════════════

def _find_masks_dir(data_dir: str) -> str:
    sub = os.path.join(data_dir, "masks")
    return sub if os.path.isdir(sub) else data_dir

def _discover_mask_samples(data_dir: str) -> List[int]:
    d = _find_masks_dir(data_dir)
    return sorted(int(m.group(1)) for name in os.listdir(d) if (m := re.match(r"sample_(\d+)\.pt$", name)))

def _load_mask_sample(data_dir: str, si: int):
    import torch
    d = _find_masks_dir(data_dir)
    return torch.load(os.path.join(d, f"sample_{si:04d}.pt"), map_location="cpu")

def _load_all_masks(data_dir: str, sample_indices=None):
    if sample_indices is None:
        sample_indices = _discover_mask_samples(data_dir)
    if not sample_indices:
        raise FileNotFoundError(f"No mask samples in {data_dir}")
    samples_data = [_load_mask_sample(data_dir, si) for si in sample_indices]
    first = samples_data[0]
    nl = len(first)
    nh = first[0]["block_mask"].shape[0]
    nb = first[0]["block_mask"].shape[1]
    return samples_data, sample_indices, nl, nh, nb


# ---- masks modes -----------------------------------------------------------

def mask_summary(data, si, nl, nh, nb, args):
    """Per-layer statistics: active blocks, density, bar_cnt."""
    total_blocks = nb * nb
    print(f"Samples={len(data)}  Layers={nl}  Heads={nh}  Blocks={nb}×{nb}\n")
    print(f"{'Layer':<6} {'Density':>8} {'DenStd':>8} {'BarMean':>8} {'BarStd':>8}")
    print("-" * 44)
    for li in range(nl):
        densities, bars = [], []
        for d in data:
            bm = d[li]["block_mask"].float()  # [H, nB, nB]
            bc = d[li]["bar_cnt"]             # [H, nB, 2]
            densities.append((bm.sum(dim=(-1, -2)) / total_blocks).mean().item())
            bars.append(bc[..., -1].float().mean().item())
        print(f"{li:<6} {np.mean(densities):8.4f} {np.std(densities):8.4f} "
              f"{np.mean(bars):8.1f} {np.std(bars):8.1f}")


def mask_density_heatmap(data, si, nl, nh, nb, args):
    """Heatmap: mean block density per (layer, head)."""
    import matplotlib.pyplot as plt
    total = nb * nb
    density = np.zeros((nl, nh))
    for d in data:
        for li in range(nl):
            bm = d[li]["block_mask"].float()
            density[li] += (bm.sum(dim=(-1, -2)) / total).numpy()
    density /= len(data)
    fig, ax = plt.subplots(figsize=(max(8, nh * 0.4), max(6, nl * 0.25)))
    im = ax.imshow(density, aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("Head"); ax.set_ylabel("Layer")
    ax.set_title("Block Mask Density (layer × head)")
    fig.colorbar(im, ax=ax, label="Density"); fig.tight_layout()
    save_or_show(fig, args.output_dir, "mask_density_heatmap")


def mask_bar_distribution(data, si, nl, nh, nb, args):
    """Box plot of total bar_cnt per layer."""
    import matplotlib.pyplot as plt
    per_layer = []
    for li in range(nl):
        vals = []
        for d in data:
            bc = d[li]["bar_cnt"][..., -1].float()  # [H, nB]
            vals.append(bc.mean().item())
        per_layer.append(vals)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(per_layer, tick_labels=list(range(nl)), showfliers=False)
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean Bar Count (per block)")
    ax.set_title("Bar Count Distribution per Layer"); ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", labelsize=7)
    save_or_show(fig, args.output_dir, "mask_bar_distribution")


def mask_block_pattern(data, si, nl, nh, nb, args):
    """Visualise the block_mask grid for one (sample, layer, head)."""
    import matplotlib.pyplot as plt
    sample_i = int(args.sample or 0)
    layer_i = int(args.layer or 0)
    head_i = int(args.head or 0)
    if sample_i >= len(data):
        sys.exit(f"Sample {sample_i} out of range (have {len(data)})")
    bm = data[sample_i][layer_i]["block_mask"][head_i].float().numpy()  # [nB, nB]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(bm, cmap="Blues", origin="upper")
    ax.set_xlabel("K block"); ax.set_ylabel("Q block")
    ax.set_title(f"Block Mask  sample={si[sample_i]} layer={layer_i} head={head_i}")
    fig.tight_layout()
    save_or_show(fig, args.output_dir, f"mask_pattern_s{si[sample_i]}_l{layer_i}_h{head_i}")


def mask_sparsity_per_layer(data, si, nl, nh, nb, args):
    """Bar chart: block-level sparsity per layer."""
    import matplotlib.pyplot as plt
    total = nb * nb
    sp = np.zeros(nl)
    for d in data:
        for li in range(nl):
            active = d[li]["block_mask"].float().sum(dim=(-1, -2)).mean().item()
            sp[li] += 1.0 - active / total
    sp /= len(data)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(nl), sp, color="teal", alpha=0.8)
    ax.set_xlabel("Layer"); ax.set_ylabel("Block Sparsity (1 − density)")
    ax.set_title("Block-level Sparsity per Layer"); ax.grid(True, alpha=0.3, axis="y")
    save_or_show(fig, args.output_dir, "mask_sparsity_per_layer")


def mask_export_csv(data, si, nl, nh, nb, args):
    out = os.path.join(args.output_dir or ".", "masks.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    total = nb * nb
    with open(out, "w") as f:
        f.write("sample,layer,head,active_blocks,total_blocks,bar_cnt_total\n")
        for i, d in enumerate(data):
            for li in range(nl):
                bm = d[li]["block_mask"]
                bc = d[li]["bar_cnt"]
                for hi in range(nh):
                    ab = int(bm[hi].sum().item())
                    bt = int(bc[hi, :, -1].sum().item())
                    f.write(f"{si[i]},{li},{hi},{ab},{total},{bt}\n")
    print(f"Exported → {out}")


def mask_draw_all(data, si, nl, nh, nb, args):
    for name, fn in [("summary", mask_summary), ("density_heatmap", mask_density_heatmap),
                     ("bar_distribution", mask_bar_distribution),
                     ("sparsity_per_layer", mask_sparsity_per_layer)]:
        print(f"  → {name}")
        fn(data, si, nl, nh, nb, args)


MASK_MODES = {
    "summary": mask_summary,
    "density_heatmap": mask_density_heatmap,
    "bar_distribution": mask_bar_distribution,
    "block_pattern": mask_block_pattern,
    "sparsity_per_layer": mask_sparsity_per_layer,
    "export_csv": mask_export_csv,
    "draw_all": mask_draw_all,
}


def run_masks(args):
    tag = args.ckpt_tag
    tags = discover_checkpoints(args.base_dir)
    if not tags:
        sys.exit(f"No checkpoints in {args.base_dir}")
    if tag is None:
        tag = tags[-1]
    data_dir = os.path.join(args.base_dir, tag)
    print(f"Using checkpoint: {tag}")
    data, si, nl, nh, nb = _load_all_masks(data_dir)
    print(f"Loaded {len(data)} samples, {nl} layers, {nh} heads, {nb}×{nb} blocks\n")
    MASK_MODES[args.mode](data, si, nl, nh, nb, args)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Unified sparse attention data analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # ---------- ratio ----------
    p_r = sub.add_parser("ratio", help="Sparse ratio trend analysis")
    p_r.add_argument("--base_dir", required=True, help="Root dir with checkpoint-tag subdirs")
    p_r.add_argument("--mode", choices=list(RATIO_MODES), default="summary")
    p_r.add_argument("--layers", default=None, help="Layer selection spec")
    p_r.add_argument("--ckpts", default=None, help="Checkpoint selection spec")
    p_r.add_argument("--output_dir", default=None, help="Directory for plots")

    # ---------- indices ----------
    p_i = sub.add_parser("indices", help="Vertical/slash index analysis")
    p_i.add_argument("--base_dir", required=True, help="Root dir with checkpoint-tag subdirs")
    p_i.add_argument("--ckpt_tag", default=None, help="Checkpoint tag (default: latest)")
    p_i.add_argument("--mode", choices=list(IDX_MODES), default="summary")
    p_i.add_argument("--layers", default=None, help="Layer selection spec")
    p_i.add_argument("--output_dir", default=None, help="Directory for plots")

    # ---------- masks ----------
    p_m = sub.add_parser("masks", help="Block mask / bar count analysis")
    p_m.add_argument("--base_dir", required=True, help="Root dir with checkpoint-tag subdirs")
    p_m.add_argument("--ckpt_tag", default=None, help="Checkpoint tag (default: latest)")
    p_m.add_argument("--mode", choices=list(MASK_MODES), default="summary")
    p_m.add_argument("--layers", default=None, help="Layer selection spec")
    p_m.add_argument("--sample", default=None, help="Sample index for block_pattern mode")
    p_m.add_argument("--layer", default=None, help="Layer index for block_pattern mode")
    p_m.add_argument("--head", default=None, help="Head index for block_pattern mode")
    p_m.add_argument("--output_dir", default=None, help="Directory for plots")

    args = parser.parse_args()
    {"ratio": run_ratio, "indices": run_indices, "masks": run_masks}[args.subcommand](args)


if __name__ == "__main__":
    main()
