#!/usr/bin/env python3
# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Flexible analysis of sparse ratio data collected during sparse training.

Data layout:
    <base_dir>/
    ├── 0000-0000/sparse_ratio/<attn_name>/sparse_ratio_rank_0.json
    ├── 0000-0001/sparse_ratio/<attn_name>/sparse_ratio_rank_0.json
    └── ...

Each JSON: { sample_idx: { layer_idx: sparse_ratio, ... }, ... }

Usage examples:
    # 1. Overview: print summary statistics across all checkpoints
    python analyze_sparse_ratio.py --base_dir <path> --mode summary

    # 2. Plot sparse ratio vs training step (averaged over samples), per layer
    python analyze_sparse_ratio.py --base_dir <path> --mode layer_over_steps

    # 3. Plot sparse ratio vs layer index, per checkpoint
    python analyze_sparse_ratio.py --base_dir <path> --mode step_over_layers

    # 4. Heatmap: layers x checkpoints
    python analyze_sparse_ratio.py --base_dir <path> --mode heatmap

    # 5. Per-sample distribution at a given checkpoint
    python analyze_sparse_ratio.py --base_dir <path> --mode sample_dist --ckpt_tag 0000-0020

    # 6. Export a CSV for downstream use
    python analyze_sparse_ratio.py --base_dir <path> --mode export_csv --output sparse_ratio.csv
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def discover_checkpoints(base_dir: str) -> List[str]:
    """Return sorted list of checkpoint tags (e.g. '0000-0000') found in base_dir."""
    tags = []
    for name in os.listdir(base_dir):
        if re.match(r"^\d{4}-\d{4}$", name):
            tags.append(name)
    return sorted(tags)


def load_sparse_ratios(
    base_dir: str,
    attn_name: str = "qwen_flex_090",
    rank: int = 0,
    ckpt_tags: Optional[List[str]] = None,
) -> Dict[str, Dict[int, Dict[int, float]]]:
    """Load sparse ratio JSONs.

    Returns:
        { ckpt_tag: { sample_idx(int): { layer_idx(int): ratio } } }
    """
    if ckpt_tags is None:
        ckpt_tags = discover_checkpoints(base_dir)

    data = {}
    for tag in ckpt_tags:
        json_path = os.path.join(
            base_dir, tag, "sparse_ratio", attn_name, f"sparse_ratio_rank_{rank}.json"
        )
        if not os.path.isfile(json_path):
            print(f"[warn] missing: {json_path}", file=sys.stderr)
            continue
        with open(json_path) as f:
            raw = json.load(f)
        data[tag] = {
            int(s): {int(l): v for l, v in layers.items()}
            for s, layers in raw.items()
        }
    return data


def to_numpy(
    data: Dict[str, Dict[int, Dict[int, float]]]
) -> Tuple[np.ndarray, List[str], List[int], List[int]]:
    """Convert loaded data to a 3-D numpy array (ckpt, sample, layer).

    Returns (array, ckpt_tags, sample_indices, layer_indices).
    Missing entries are NaN.
    """
    ckpt_tags = sorted(data.keys())
    all_samples = sorted({s for d in data.values() for s in d})
    all_layers = sorted({l for d in data.values() for sd in d.values() for l in sd})

    arr = np.full((len(ckpt_tags), len(all_samples), len(all_layers)), np.nan)
    s_idx_map = {s: i for i, s in enumerate(all_samples)}
    l_idx_map = {l: i for i, l in enumerate(all_layers)}

    for ci, tag in enumerate(ckpt_tags):
        for s, layers in data[tag].items():
            for l, v in layers.items():
                arr[ci, s_idx_map[s], l_idx_map[l]] = v

    return arr, ckpt_tags, all_samples, all_layers


def ckpt_tag_to_step(tag: str) -> int:
    """Convert '0000-0039' -> 39 (epoch * 10000 + iter)."""
    parts = tag.split("-")
    return int(parts[0]) * 10000 + int(parts[1])


# ---------------------------------------------------------------------------
# Analysis modes
# ---------------------------------------------------------------------------


def mode_summary(arr, ckpt_tags, samples, layers, args):
    """Print overall statistics."""
    print(f"Checkpoints : {len(ckpt_tags)} ({ckpt_tags[0]} .. {ckpt_tags[-1]})")
    print(f"Samples/ckpt: {len(samples)}")
    print(f"Layers      : {len(layers)}")
    print()

    # Per-checkpoint stats (mean over samples & layers)
    ckpt_means = np.nanmean(arr, axis=(1, 2))
    print(f"{'Checkpoint':<14} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 50)
    for i, tag in enumerate(ckpt_tags):
        vals = arr[i]
        print(
            f"{tag:<14} {np.nanmean(vals):8.4f} {np.nanstd(vals):8.4f} "
            f"{np.nanmin(vals):8.4f} {np.nanmax(vals):8.4f}"
        )
    print("-" * 50)
    print(
        f"{'Overall':<14} {np.nanmean(arr):8.4f} {np.nanstd(arr):8.4f} "
        f"{np.nanmin(arr):8.4f} {np.nanmax(arr):8.4f}"
    )

    # Per-layer stats (mean over ckpts & samples)
    print()
    print(f"{'Layer':<8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 44)
    for j, l in enumerate(layers):
        vals = arr[:, :, j]
        print(
            f"{l:<8} {np.nanmean(vals):8.4f} {np.nanstd(vals):8.4f} "
            f"{np.nanmin(vals):8.4f} {np.nanmax(vals):8.4f}"
        )


def mode_layer_over_steps(arr, ckpt_tags, samples, layers, args):
    """Plot: sparse ratio vs training step, one line per layer (mean over samples)."""
    import matplotlib.pyplot as plt

    steps = [ckpt_tag_to_step(t) for t in ckpt_tags]
    # arr shape: (ckpt, sample, layer) -> mean over samples -> (ckpt, layer)
    means = np.nanmean(arr, axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    layer_indices = list(range(len(layers)))

    # If many layers, show a subset or use colormap
    show_layers = _select_layers(layers, args.layers)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(show_layers)))

    for idx, li in enumerate(show_layers):
        j = layers.index(li)
        ax.plot(steps, means[:, j], label=f"Layer {li}", color=cmap[idx], alpha=0.8)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Sparse Ratio")
    ax.set_title("Sparse Ratio vs Training Step (per layer, mean over samples)")
    ax.legend(fontsize="small", ncol=max(1, len(show_layers) // 12), loc="best")
    ax.grid(True, alpha=0.3)
    _save_or_show(fig, args.output, "layer_over_steps")


def mode_step_over_layers(arr, ckpt_tags, samples, layers, args):
    """Plot: sparse ratio vs layer index, one line per checkpoint (mean over samples)."""
    import matplotlib.pyplot as plt

    means = np.nanmean(arr, axis=1)  # (ckpt, layer)

    fig, ax = plt.subplots(figsize=(12, 6))
    show_ckpts = _select_ckpts(ckpt_tags, args.ckpts)
    cmap = plt.cm.coolwarm(np.linspace(0, 1, len(show_ckpts)))

    for idx, tag in enumerate(show_ckpts):
        i = ckpt_tags.index(tag)
        ax.plot(layers, means[i, :], label=tag, color=cmap[idx], alpha=0.8)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Sparse Ratio")
    ax.set_title("Sparse Ratio vs Layer (per checkpoint, mean over samples)")
    ax.legend(fontsize="small", ncol=max(1, len(show_ckpts) // 8), loc="best")
    ax.grid(True, alpha=0.3)
    _save_or_show(fig, args.output, "step_over_layers")


def mode_heatmap(arr, ckpt_tags, samples, layers, args):
    """Heatmap: layers (x) vs checkpoints (y), mean over samples."""
    import matplotlib
    import matplotlib.pyplot as plt

    means = np.nanmean(arr, axis=1)  # (ckpt, layer)

    fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.35), max(6, len(ckpt_tags) * 0.25)))
    im = ax.imshow(means, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=7)
    ax.set_yticks(range(len(ckpt_tags)))
    ax.set_yticklabels(ckpt_tags, fontsize=7)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Checkpoint")
    ax.set_title("Sparse Ratio Heatmap (mean over samples)")
    fig.colorbar(im, ax=ax, label="Sparse Ratio")
    fig.tight_layout()
    _save_or_show(fig, args.output, "heatmap")


def mode_sample_dist(arr, ckpt_tags, samples, layers, args):
    """Box/violin plot of per-sample sparse ratios at a single checkpoint."""
    import matplotlib.pyplot as plt

    tag = args.ckpt_tag or ckpt_tags[-1]
    if tag not in ckpt_tags:
        print(f"Checkpoint '{tag}' not found. Available: {ckpt_tags}", file=sys.stderr)
        sys.exit(1)
    ci = ckpt_tags.index(tag)
    data_slice = arr[ci]  # (sample, layer)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Per-layer distribution across samples
    ax = axes[0]
    bp = ax.boxplot(
        [data_slice[:, j] for j in range(len(layers))],
        tick_labels=layers,
        showfliers=False,
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Sparse Ratio")
    ax.set_title(f"Per-layer sample distribution @ {tag}")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # Per-sample mean distribution
    ax = axes[1]
    sample_means = np.nanmean(data_slice, axis=1)
    ax.bar(range(len(samples)), sample_means, color="steelblue", alpha=0.8)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Mean Sparse Ratio (across layers)")
    ax.set_title(f"Per-sample mean @ {tag}")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Sample Distribution at Checkpoint {tag}", fontsize=13)
    fig.tight_layout()
    _save_or_show(fig, args.output, f"sample_dist_{tag}")


def mode_layer_trend(arr, ckpt_tags, samples, layers, args):
    """For selected layers, plot mean ± std over samples across checkpoints."""
    import matplotlib.pyplot as plt

    steps = [ckpt_tag_to_step(t) for t in ckpt_tags]
    show_layers = _select_layers(layers, args.layers)

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.tab10(np.linspace(0, 1, min(len(show_layers), 10)))

    for idx, li in enumerate(show_layers):
        j = layers.index(li)
        layer_vals = arr[:, :, j]  # (ckpt, sample)
        mu = np.nanmean(layer_vals, axis=1)
        sigma = np.nanstd(layer_vals, axis=1)
        color = cmap[idx % len(cmap)]
        ax.plot(steps, mu, label=f"Layer {li}", color=color)
        ax.fill_between(steps, mu - sigma, mu + sigma, color=color, alpha=0.15)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Sparse Ratio")
    ax.set_title("Layer Trend: mean ± std over samples")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)
    _save_or_show(fig, args.output, "layer_trend")


def mode_compare_attn(arr_dict, ckpt_tags_dict, layers_dict, args):
    """Compare multiple attn configs side-by-side (global mean over steps)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, arr in arr_dict.items():
        tags = ckpt_tags_dict[name]
        _layers = layers_dict[name]
        steps = [ckpt_tag_to_step(t) for t in tags]
        global_mean = np.nanmean(arr, axis=(1, 2))  # (ckpt,)
        ax.plot(steps, global_mean, label=name, marker="o", markersize=3)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Sparse Ratio")
    ax.set_title("Sparse Ratio Comparison Across Attention Configs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_or_show(fig, args.output, "compare_attn")


def mode_global_trend(arr, ckpt_tags, samples, layers, args):
    """Plot global mean sparse ratio over training steps with min/max band."""
    import matplotlib.pyplot as plt

    steps = [ckpt_tag_to_step(t) for t in ckpt_tags]
    # Per-checkpoint: mean, min, max across all samples and layers
    global_mean = np.nanmean(arr, axis=(1, 2))
    global_min = np.nanmin(arr.reshape(len(ckpt_tags), -1), axis=1)
    global_max = np.nanmax(arr.reshape(len(ckpt_tags), -1), axis=1)
    global_std = np.nanstd(arr, axis=(1, 2))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, global_mean, color="steelblue", linewidth=2, label="Mean")
    ax.fill_between(steps, global_mean - global_std, global_mean + global_std,
                    color="steelblue", alpha=0.2, label="±1 std")
    ax.fill_between(steps, global_min, global_max,
                    color="steelblue", alpha=0.07, label="Min–Max")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Sparse Ratio")
    ax.set_title("Global Sparse Ratio Trend Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_or_show(fig, args.output, "global_trend")


def mode_layer_histogram(arr, ckpt_tags, samples, layers, args):
    """Histogram of sparse ratios across all layers at selected checkpoints."""
    import matplotlib.pyplot as plt

    show_ckpts = _select_ckpts(ckpt_tags, args.ckpts)
    if len(show_ckpts) > 8:
        show_ckpts = show_ckpts[:: max(1, len(show_ckpts) // 8)]

    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.cm.coolwarm(np.linspace(0, 1, len(show_ckpts)))
    bins = np.linspace(0.3, 1.0, 50)

    for idx, tag in enumerate(show_ckpts):
        ci = ckpt_tags.index(tag)
        vals = arr[ci].flatten()
        vals = vals[~np.isnan(vals)]
        ax.hist(vals, bins=bins, alpha=0.45, color=cmap[idx], label=tag, histtype="stepfilled", edgecolor="none")

    ax.set_xlabel("Sparse Ratio")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Sparse Ratios Across Layers")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3, axis="y")
    _save_or_show(fig, args.output, "layer_histogram")


def mode_per_layer_grid(arr, ckpt_tags, samples, layers, args):
    """Small-multiples grid: one subplot per layer showing trend over steps."""
    import matplotlib.pyplot as plt

    steps = np.array([ckpt_tag_to_step(t) for t in ckpt_tags])
    n_layers = len(layers)
    ncols = min(6, n_layers)
    nrows = (n_layers + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.5 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for j, l in enumerate(layers):
        r, c = divmod(j, ncols)
        ax = axes[r][c]
        layer_vals = arr[:, :, j]  # (ckpt, sample)
        mu = np.nanmean(layer_vals, axis=1)
        lo = np.nanpercentile(layer_vals, 10, axis=1)
        hi = np.nanpercentile(layer_vals, 90, axis=1)
        ax.plot(steps, mu, color="steelblue", linewidth=1.2)
        ax.fill_between(steps, lo, hi, color="steelblue", alpha=0.15)
        ax.set_title(f"Layer {l}", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    # Hide unused subplots
    for j in range(n_layers, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    fig.supxlabel("Training Step", fontsize=10)
    fig.supylabel("Sparse Ratio", fontsize=10)
    fig.suptitle("Per-Layer Sparse Ratio Over Training (10th–90th percentile)", fontsize=12)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.96])
    _save_or_show(fig, args.output, "per_layer_grid")


def mode_delta_heatmap(arr, ckpt_tags, samples, layers, args):
    """Heatmap of sparse ratio *change* from first checkpoint, per layer."""
    import matplotlib.pyplot as plt

    means = np.nanmean(arr, axis=1)  # (ckpt, layer)
    delta = means - means[0:1, :]   # relative to first checkpoint

    vabs = max(abs(np.nanmin(delta)), abs(np.nanmax(delta)), 0.01)
    fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.35), max(6, len(ckpt_tags) * 0.25)))
    im = ax.imshow(delta, aspect="auto", cmap="RdBu_r", vmin=-vabs, vmax=vabs)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=7)
    ax.set_yticks(range(len(ckpt_tags)))
    ax.set_yticklabels(ckpt_tags, fontsize=7)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Checkpoint")
    ax.set_title(f"Sparse Ratio Change from {ckpt_tags[0]} (mean over samples)")
    fig.colorbar(im, ax=ax, label="\u0394 Sparse Ratio")
    fig.tight_layout()
    _save_or_show(fig, args.output, "delta_heatmap")


def mode_export_csv(arr, ckpt_tags, samples, layers, args):
    """Export all data to a flat CSV."""
    output_path = args.output or "sparse_ratio_export.csv"
    with open(output_path, "w") as f:
        f.write("checkpoint,step,sample,layer,sparse_ratio\n")
        for ci, tag in enumerate(ckpt_tags):
            step = ckpt_tag_to_step(tag)
            for si, s in enumerate(samples):
                for li, l in enumerate(layers):
                    v = arr[ci, si, li]
                    if not np.isnan(v):
                        f.write(f"{tag},{step},{s},{l},{v:.6f}\n")
    print(f"Exported to {output_path}")


def mode_draw_all(arr, ckpt_tags, samples, layers, args):
    """Generate all figure types into the output directory."""
    if not args.output:
        print("--output directory is required for draw_all mode.", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.output, exist_ok=True)
    print(f"Generating all figures into {args.output}/\n")

    figure_modes = [
        ("global_trend",     mode_global_trend),
        ("heatmap",          mode_heatmap),
        ("delta_heatmap",    mode_delta_heatmap),
        ("layer_over_steps", mode_layer_over_steps),
        ("step_over_layers", mode_step_over_layers),
        ("layer_trend",      mode_layer_trend),
        ("layer_histogram",  mode_layer_histogram),
        ("per_layer_grid",   mode_per_layer_grid),
    ]

    # Also generate sample_dist for first, middle, and last checkpoint
    dist_tags = [ckpt_tags[0], ckpt_tags[len(ckpt_tags) // 2], ckpt_tags[-1]]

    for name, fn in figure_modes:
        print(f"  Drawing {name}...")
        fn(arr, ckpt_tags, samples, layers, args)

    for tag in dist_tags:
        print(f"  Drawing sample_dist @ {tag}...")
        import copy
        sub_args = copy.copy(args)
        sub_args.ckpt_tag = tag
        mode_sample_dist(arr, ckpt_tags, samples, layers, sub_args)

    print(f"\nDone. All figures saved to {args.output}/")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_layers(all_layers: List[int], spec: Optional[str]) -> List[int]:
    """Parse layer selection spec: 'all', '0,5,10', '0-35:5' (start-end:step), or None (auto)."""
    if spec is None or spec == "auto":
        n = len(all_layers)
        if n <= 12:
            return all_layers
        step = max(1, n // 12)
        return all_layers[::step]
    if spec == "all":
        return list(all_layers)
    if ":" in spec or "-" in spec.split(",")[0]:
        # Range spec like '0-35:5'
        m = re.match(r"(\d+)-(\d+)(?::(\d+))?", spec)
        if m:
            start, end, step = int(m.group(1)), int(m.group(2)), int(m.group(3) or 1)
            return [l for l in all_layers if start <= l <= end and (l - start) % step == 0]
    # Comma-separated
    return [int(x) for x in spec.split(",") if int(x) in all_layers]


def _select_ckpts(all_tags: List[str], spec: Optional[str]) -> List[str]:
    """Parse checkpoint selection spec: 'all', '0000-0000,0000-0020', or None (auto)."""
    if spec is None or spec == "auto":
        n = len(all_tags)
        if n <= 10:
            return all_tags
        step = max(1, n // 10)
        return all_tags[::step]
    if spec == "all":
        return list(all_tags)
    return [t.strip() for t in spec.split(",") if t.strip() in all_tags]


def _save_or_show(fig, output: Optional[str], default_name: str):
    import matplotlib.pyplot as plt

    if output:
        path = output if output.endswith((".png", ".pdf", ".svg")) else f"{output}/{default_name}.png"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MODES = {
    "summary": "Print summary statistics",
    "layer_over_steps": "Plot sparse ratio vs step (per layer)",
    "step_over_layers": "Plot sparse ratio vs layer (per checkpoint)",
    "heatmap": "Heatmap: layers x checkpoints",
    "delta_heatmap": "Heatmap of sparse ratio change from first checkpoint",
    "sample_dist": "Per-sample distribution at one checkpoint",
    "layer_trend": "Layer trend: mean ± std over samples",
    "global_trend": "Global mean sparse ratio over training",
    "layer_histogram": "Histogram of sparse ratios at selected checkpoints",
    "per_layer_grid": "Small-multiples: per-layer trend over training",
    "compare_attn": "Compare multiple attention configs",
    "export_csv": "Export all data to CSV",
    "draw_all": "Generate ALL figure types into output directory",
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Analyze sparse ratio data from sparse training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k:<22} {v}" for k, v in MODES.items()),
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/blob/mtrain_expr_data_store/A100_32/mtrain_qwen/qwen_3B_fp090_512K_sr",
        help="Root directory containing checkpoint folders.",
    )
    parser.add_argument(
        "--attn_name",
        type=str,
        default="qwen_flex_090",
        help="Attention config name (subdirectory under sparse_ratio/).",
    )
    parser.add_argument(
        "--attn_names",
        type=str,
        default=None,
        help="Comma-separated attn names for compare_attn mode.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Rank index for the sparse ratio file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=list(MODES.keys()),
        default="summary",
        help="Analysis mode.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Layer selection: 'all', '0,5,10', '0-35:5', or 'auto' (default).",
    )
    parser.add_argument(
        "--ckpts",
        type=str,
        default=None,
        help="Checkpoint selection: 'all', '0000-0000,0000-0020', or 'auto' (default).",
    )
    parser.add_argument(
        "--ckpt_tag",
        type=str,
        default=None,
        help="Single checkpoint tag for sample_dist mode.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path: file path for CSV/image, or directory for plots.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "compare_attn":
        attn_names = (args.attn_names or args.attn_name).split(",")
        arr_dict, tags_dict, layers_dict = {}, {}, {}
        for name in attn_names:
            name = name.strip()
            data = load_sparse_ratios(args.base_dir, attn_name=name, rank=args.rank)
            if not data:
                print(f"[warn] No data for attn_name={name}", file=sys.stderr)
                continue
            arr, tags, samples, layers = to_numpy(data)
            arr_dict[name] = arr
            tags_dict[name] = tags
            layers_dict[name] = layers
        if arr_dict:
            mode_compare_attn(arr_dict, tags_dict, layers_dict, args)
        return

    data = load_sparse_ratios(args.base_dir, attn_name=args.attn_name, rank=args.rank)
    if not data:
        print("No data found.", file=sys.stderr)
        sys.exit(1)

    arr, ckpt_tags, samples, layers = to_numpy(data)
    print(f"Loaded: {arr.shape[0]} checkpoints, {arr.shape[1]} samples, {arr.shape[2]} layers\n")

    dispatch = {
        "summary": mode_summary,
        "layer_over_steps": mode_layer_over_steps,
        "step_over_layers": mode_step_over_layers,
        "heatmap": mode_heatmap,
        "delta_heatmap": mode_delta_heatmap,
        "sample_dist": mode_sample_dist,
        "layer_trend": mode_layer_trend,
        "global_trend": mode_global_trend,
        "layer_histogram": mode_layer_histogram,
        "per_layer_grid": mode_per_layer_grid,
        "export_csv": mode_export_csv,
        "draw_all": mode_draw_all,
    }
    dispatch[args.mode](arr, ckpt_tags, samples, layers, args)


if __name__ == "__main__":
    main()
