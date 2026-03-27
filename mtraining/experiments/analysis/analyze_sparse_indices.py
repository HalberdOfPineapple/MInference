#!/usr/bin/env python3
# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Analysis of sparse attention indices collected during single-card inference.

Data layout (produced by ``infer_sparse_indices.py``)::

    <data_dir>/
    ├── sample_0000.pt
    ├── sample_0001.pt
    └── ...

Each ``.pt`` file: ``{layer_idx -> {"v_idx": Tensor[H, V], "s_idx": Tensor[H, S]}}``

- ``v_idx`` values: token positions (int32); padding = 2147483647
- ``s_idx`` values: diagonal positions (int32); padding = -1

Usage examples::

    # 1. Print summary statistics
    python analyze_sparse_indices.py --data_dir <path> --mode summary

    # 2. Count of valid (non-padding) indices per head/layer heatmap
    python analyze_sparse_indices.py --data_dir <path> --mode count_heatmap

    # 3. Distribution of vertical index positions
    python analyze_sparse_indices.py --data_dir <path> --mode position_dist

    # 4. Cross-sample overlap (Jaccard similarity) of selected indices
    python analyze_sparse_indices.py --data_dir <path> --mode overlap

    # 5. Per-layer sparsity ratio (derived from indices)
    python analyze_sparse_indices.py --data_dir <path> --mode sparsity

    # 6. Export flattened CSV for downstream tools
    python analyze_sparse_indices.py --data_dir <path> --mode export_csv --output indices.csv

    # 7. Run all visual analyses and save PNGs
    python analyze_sparse_indices.py --data_dir <path> --mode draw_all --output_dir plots/

    # ---- Discreteness analysis ----

    # 8. Quantitative gap/run/fragmentation table per layer
    python analyze_sparse_indices.py --data_dir <path> --mode discreteness

    # 9. Histogram of inter-index gaps (log-scale) per layer
    python analyze_sparse_indices.py --data_dir <path> --mode gap_histogram

    # 10. Fragmentation index & mean run length across layers
    python analyze_sparse_indices.py --data_dir <path> --mode fragmentation

    # 11. Raster (spike) plot of selected positions per head
    python analyze_sparse_indices.py --data_dir <path> --mode raster --layers 0,17,35

    # 12. Binned spatial density heatmap (layers × sequence regions)
    python analyze_sparse_indices.py --data_dir <path> --mode density_heatmap

    # ---- Cross-iteration dynamics (require --base_dir) ----

    # 13. Fragmentation index vs training step, per layer
    python analyze_sparse_indices.py --base_dir <path> --mode frag_over_steps

    # 14. Valid index count vs training step, per layer
    python analyze_sparse_indices.py --base_dir <path> --mode count_over_steps

    # 15. Heatmap: layers × training steps, coloured by fragmentation
    python analyze_sparse_indices.py --base_dir <path> --mode dynamics_heatmap

    # 16. Jaccard similarity between consecutive checkpoints
    python analyze_sparse_indices.py --base_dir <path> --mode overlap_over_steps
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

V_IDX_PAD = 2147483647  # INT32_MAX — padding for unused vertical slots
S_IDX_PAD = -1  # padding for unused slash slots


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def discover_samples(data_dir: str) -> List[int]:
    """Return sorted list of sample indices found in data_dir."""
    indices = []
    for name in os.listdir(data_dir):
        m = re.match(r"^sample_(\d+)\.pt$", name)
        if m:
            indices.append(int(m.group(1)))
    return sorted(indices)


def load_sample(data_dir: str, sample_idx: int) -> Dict[int, Dict[str, torch.Tensor]]:
    """Load a single sample's index data.

    Returns: {layer_idx -> {"v_idx": Tensor[H, V], "s_idx": Tensor[H, S]}}
    """
    path = os.path.join(data_dir, f"sample_{sample_idx:04d}.pt")
    return torch.load(path, map_location="cpu")


def load_all(data_dir: str, sample_indices: Optional[List[int]] = None):
    """Load all (or selected) samples.

    Returns:
        samples_data: list of per-sample dicts
        sample_indices: list of sample index ints
        num_layers: int
        num_heads: int
    """
    if sample_indices is None:
        sample_indices = discover_samples(data_dir)
    if not sample_indices:
        raise FileNotFoundError(f"No sample_*.pt files found in {data_dir}")

    samples_data = []
    for si in sample_indices:
        samples_data.append(load_sample(data_dir, si))

    # Infer dimensions from first sample
    first = samples_data[0]
    num_layers = len(first)
    num_heads = first[0]["v_idx"].shape[0]
    return samples_data, sample_indices, num_layers, num_heads


def count_valid(tensor: torch.Tensor, pad_value: int) -> torch.Tensor:
    """Count non-padding entries along the last dim.  Returns shape [H]."""
    return (tensor != pad_value).sum(dim=-1)


# ---------------------------------------------------------------------------
# Multi-iteration data loading
# ---------------------------------------------------------------------------


def discover_checkpoints(base_dir: str) -> List[str]:
    """Return sorted list of checkpoint tags (e.g. '0000-0001') found in base_dir.

    Each tag must be a subdirectory containing at least one ``sample_*.pt`` file.
    """
    tags = []
    for name in sorted(os.listdir(base_dir)):
        if re.match(r"^\d{4}-\d{4}$", name):
            subdir = os.path.join(base_dir, name)
            if os.path.isdir(subdir) and discover_samples(subdir):
                tags.append(name)
    return sorted(tags)


def ckpt_tag_to_step(tag: str) -> int:
    """Convert '0000-0039' -> 39 (epoch * 10000 + iter)."""
    parts = tag.split("-")
    return int(parts[0]) * 10000 + int(parts[1])


def load_multi_iter(
    base_dir: str,
    ckpt_tags: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
):
    """Load index data across multiple checkpoint iterations.

    Args:
        base_dir: root directory containing checkpoint-tag subdirectories,
            each with ``sample_*.pt`` files.
        ckpt_tags: list of tags to load; ``None`` = auto-discover.
        max_samples: limit samples loaded per checkpoint (for speed).

    Returns:
        multi_data: dict  {ckpt_tag -> list of per-sample dicts}
        ckpt_tags: sorted list of tags
        num_layers, num_heads: int
    """
    if ckpt_tags is None:
        ckpt_tags = discover_checkpoints(base_dir)
    if not ckpt_tags:
        raise FileNotFoundError(f"No checkpoint subdirs found in {base_dir}")

    multi_data: Dict[str, list] = {}
    num_layers = num_heads = 0
    for tag in ckpt_tags:
        data_dir = os.path.join(base_dir, tag)
        si = discover_samples(data_dir)
        if max_samples is not None:
            si = si[:max_samples]
        samples_data, _, nl, nh = load_all(data_dir, si)
        multi_data[tag] = samples_data
        num_layers, num_heads = nl, nh
    return multi_data, ckpt_tags, num_layers, num_heads


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_layers(num_layers: int, spec: str) -> List[int]:
    """Parse layer selection spec.  Supports 'all', '0,5,10', '0-35:5', 'auto'."""
    if spec == "all":
        return list(range(num_layers))
    if spec == "auto":
        if num_layers <= 12:
            return list(range(num_layers))
        step = max(1, num_layers // 12)
        return list(range(0, num_layers, step))
    if "-" in spec and ":" in spec:
        rng, step = spec.rsplit(":", 1)
        lo, hi = rng.split("-")
        return list(range(int(lo), int(hi) + 1, int(step)))
    return [int(x) for x in spec.split(",")]


def _select_heads(num_heads: int, spec: str) -> List[int]:
    """Parse head selection spec (same syntax as layers)."""
    return _select_layers(num_heads, spec)


def _save_or_show(fig, output: Optional[str]):
    """Save figure to file or show interactively."""
    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved: {output}")
    else:
        import matplotlib.pyplot as plt
        plt.show()
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis: summary
# ---------------------------------------------------------------------------


def mode_summary(samples_data, sample_indices, num_layers, num_heads, args):
    """Print overall statistics about collected indices."""
    num_samples = len(samples_data)
    print(f"Samples : {num_samples}")
    print(f"Layers  : {num_layers}")
    print(f"Heads   : {num_heads}")

    # Collect valid counts: shape (num_samples, num_layers, num_heads)
    v_counts = np.zeros((num_samples, num_layers, num_heads))
    s_counts = np.zeros((num_samples, num_layers, num_heads))

    for si, data in enumerate(samples_data):
        for li in range(num_layers):
            v_counts[si, li] = count_valid(data[li]["v_idx"], V_IDX_PAD).numpy()
            s_counts[si, li] = count_valid(data[li]["s_idx"], S_IDX_PAD).numpy()

    # Max allocated size (tensor width)
    max_v_alloc = samples_data[0][0]["v_idx"].shape[-1]
    max_s_alloc = samples_data[0][0]["s_idx"].shape[-1]

    print(f"\nVertical indices (max alloc = {max_v_alloc}):")
    print(f"  Mean valid/head : {v_counts.mean():.1f}")
    print(f"  Min  valid/head : {v_counts.min():.0f}")
    print(f"  Max  valid/head : {v_counts.max():.0f}")

    print(f"\nSlash indices (max alloc = {max_s_alloc}):")
    print(f"  Mean valid/head : {s_counts.mean():.1f}")
    print(f"  Min  valid/head : {s_counts.min():.0f}")
    print(f"  Max  valid/head : {s_counts.max():.0f}")

    # Per-layer breakdown
    print(f"\n{'Layer':<8} {'V mean':>8} {'V std':>8} {'S mean':>8} {'S std':>8}")
    print("-" * 44)
    for li in range(num_layers):
        vm = v_counts[:, li, :].mean()
        vs = v_counts[:, li, :].std()
        sm = s_counts[:, li, :].mean()
        ss = s_counts[:, li, :].std()
        print(f"{li:<8} {vm:8.1f} {vs:8.1f} {sm:8.1f} {ss:8.1f}")


# ---------------------------------------------------------------------------
# Analysis: count_heatmap
# ---------------------------------------------------------------------------


def mode_count_heatmap(samples_data, sample_indices, num_layers, num_heads, args):
    """Heatmap of mean valid-index count per (layer, head), averaged over samples."""
    import matplotlib.pyplot as plt

    v_counts = np.zeros((len(samples_data), num_layers, num_heads))
    s_counts = np.zeros((len(samples_data), num_layers, num_heads))
    for si, data in enumerate(samples_data):
        for li in range(num_layers):
            v_counts[si, li] = count_valid(data[li]["v_idx"], V_IDX_PAD).numpy()
            s_counts[si, li] = count_valid(data[li]["s_idx"], S_IDX_PAD).numpy()

    v_mean = v_counts.mean(axis=0)  # (layers, heads)
    s_mean = s_counts.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, num_layers * 0.25)))
    for ax, mat, title in [
        (axes[0], v_mean, "Vertical index count"),
        (axes[1], s_mean, "Slash index count"),
    ]:
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(f"{title} (mean over {len(samples_data)} samples)")
        fig.colorbar(im, ax=ax, shrink=0.6)

    fig.tight_layout()
    out = args.output or (
        os.path.join(args.output_dir, "count_heatmap.png") if args.output_dir else None
    )
    _save_or_show(fig, out)


# ---------------------------------------------------------------------------
# Analysis: position_dist
# ---------------------------------------------------------------------------


def mode_position_dist(samples_data, sample_indices, num_layers, num_heads, args):
    """Histogram of where vertical / slash indices fall in the sequence."""
    import matplotlib.pyplot as plt

    show_layers = _select_layers(num_layers, args.layers)

    # Collect all valid vertical positions across samples for selected layers
    fig, axes = plt.subplots(len(show_layers), 2, figsize=(14, 3 * len(show_layers)),
                             squeeze=False)

    for row, li in enumerate(show_layers):
        v_positions = []
        s_positions = []
        for data in samples_data:
            v = data[li]["v_idx"].numpy().flatten()
            v_positions.append(v[v != V_IDX_PAD])
            s = data[li]["s_idx"].numpy().flatten()
            s_positions.append(s[s != S_IDX_PAD])

        v_all = np.concatenate(v_positions) if v_positions else np.array([])
        s_all = np.concatenate(s_positions) if s_positions else np.array([])

        ax_v = axes[row, 0]
        if len(v_all) > 0:
            ax_v.hist(v_all, bins=100, alpha=0.7, color="steelblue", edgecolor="none")
        ax_v.set_title(f"Layer {li} — Vertical positions")
        ax_v.set_xlabel("Token position")
        ax_v.set_ylabel("Count")
        ax_v.grid(True, alpha=0.3)

        ax_s = axes[row, 1]
        if len(s_all) > 0:
            ax_s.hist(s_all, bins=100, alpha=0.7, color="coral", edgecolor="none")
        ax_s.set_title(f"Layer {li} — Slash positions")
        ax_s.set_xlabel("Diagonal position")
        ax_s.set_ylabel("Count")
        ax_s.grid(True, alpha=0.3)

    fig.suptitle("Distribution of selected index positions", fontsize=14, y=1.01)
    fig.tight_layout()
    out = args.output or (
        os.path.join(args.output_dir, "position_dist.png") if args.output_dir else None
    )
    _save_or_show(fig, out)


# ---------------------------------------------------------------------------
# Analysis: overlap  (cross-sample Jaccard similarity)
# ---------------------------------------------------------------------------


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def mode_overlap(samples_data, sample_indices, num_layers, num_heads, args):
    """Cross-sample Jaccard similarity of selected indices, per layer (mean over heads)."""
    import matplotlib.pyplot as plt

    if len(samples_data) < 2:
        print("Need at least 2 samples for overlap analysis.")
        return

    show_layers = _select_layers(num_layers, args.layers)

    # For each layer, compute pairwise Jaccard between all sample pairs, per head
    v_jaccard_per_layer = []
    s_jaccard_per_layer = []

    for li in show_layers:
        v_jaccards = []
        s_jaccards = []
        for i in range(len(samples_data)):
            for j in range(i + 1, len(samples_data)):
                for h in range(num_heads):
                    vi = samples_data[i][li]["v_idx"][h].numpy()
                    vi = vi[vi != V_IDX_PAD]
                    vj = samples_data[j][li]["v_idx"][h].numpy()
                    vj = vj[vj != V_IDX_PAD]
                    v_jaccards.append(_jaccard(vi, vj))

                    si = samples_data[i][li]["s_idx"][h].numpy()
                    si = si[si != S_IDX_PAD]
                    sj = samples_data[j][li]["s_idx"][h].numpy()
                    sj = sj[sj != S_IDX_PAD]
                    s_jaccards.append(_jaccard(si, sj))

        v_jaccard_per_layer.append(np.mean(v_jaccards))
        s_jaccard_per_layer.append(np.mean(s_jaccards))

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(show_layers))
    w = 0.35
    ax.bar(x - w / 2, v_jaccard_per_layer, w, label="Vertical", color="steelblue", alpha=0.8)
    ax.bar(x + w / 2, s_jaccard_per_layer, w, label="Slash", color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in show_layers])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Jaccard similarity")
    ax.set_title(f"Cross-sample index overlap ({len(samples_data)} samples, mean over heads)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    out = args.output or (
        os.path.join(args.output_dir, "overlap.png") if args.output_dir else None
    )
    _save_or_show(fig, out)


# ---------------------------------------------------------------------------
# Analysis: sparsity  (derive sparse ratio from indices)
# ---------------------------------------------------------------------------


def mode_sparsity(samples_data, sample_indices, num_layers, num_heads, args):
    """Compute and plot sparsity ratio derived from index counts.

    Sparsity ≈ 1 − (v_count + s_count) / seq_len  per head, averaged over heads.
    """
    import matplotlib.pyplot as plt

    # Infer sequence length from the max valid vertical index
    seq_len = 0
    for data in samples_data:
        for li in range(num_layers):
            v = data[li]["v_idx"].numpy()
            valid = v[v != V_IDX_PAD]
            if len(valid) > 0:
                seq_len = max(seq_len, int(valid.max()) + 1)
    if seq_len == 0:
        print("Cannot determine sequence length — no valid indices found.")
        return
    print(f"Inferred sequence length: {seq_len}")

    # Compute per (sample, layer) sparsity (mean over heads)
    sparsity = np.zeros((len(samples_data), num_layers))
    for si, data in enumerate(samples_data):
        for li in range(num_layers):
            vc = count_valid(data[li]["v_idx"], V_IDX_PAD).float().numpy()
            sc = count_valid(data[li]["s_idx"], S_IDX_PAD).float().numpy()
            ratio_per_head = 1.0 - (vc + sc) / seq_len
            sparsity[si, li] = ratio_per_head.mean()

    mean_sp = sparsity.mean(axis=0)
    std_sp = sparsity.std(axis=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    layers = np.arange(num_layers)
    ax.plot(layers, mean_sp, color="teal", linewidth=2, label="Mean sparsity")
    ax.fill_between(layers, mean_sp - std_sp, mean_sp + std_sp,
                     alpha=0.2, color="teal", label="±1 std")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Sparsity ratio")
    ax.set_title(f"Index-derived sparsity ratio (mean over {len(samples_data)} samples & heads)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, num_layers - 1)

    fig.tight_layout()
    out = args.output or (
        os.path.join(args.output_dir, "sparsity.png") if args.output_dir else None
    )
    _save_or_show(fig, out)


# ---------------------------------------------------------------------------
# Analysis: head_variance
# ---------------------------------------------------------------------------


def mode_head_variance(samples_data, sample_indices, num_layers, num_heads, args):
    """Box plot of valid-index counts across heads for each layer,
    showing how much heads differ in their sparsity patterns."""
    import matplotlib.pyplot as plt

    show_layers = _select_layers(num_layers, args.layers)

    # Collect counts: (samples, heads) for each layer
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, idx_type, pad, color in [
        (axes[0], "v_idx", V_IDX_PAD, "steelblue"),
        (axes[1], "s_idx", S_IDX_PAD, "coral"),
    ]:
        box_data = []
        labels = []
        for li in show_layers:
            counts = []
            for data in samples_data:
                c = count_valid(data[li][idx_type], pad).numpy()
                counts.extend(c.tolist())
            box_data.append(counts)
            labels.append(str(li))

        bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Valid index count per head")
        ax.set_title(f"{'Vertical' if idx_type == 'v_idx' else 'Slash'} count distribution")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Head variance across {len(samples_data)} samples", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = args.output or (
        os.path.join(args.output_dir, "head_variance.png") if args.output_dir else None
    )
    _save_or_show(fig, out)


# ---------------------------------------------------------------------------
# Analysis: per_head_grid
# ---------------------------------------------------------------------------


def mode_per_head_grid(samples_data, sample_indices, num_layers, num_heads, args):
    """Small-multiples grid: one subplot per head showing valid-index count across layers."""
    import matplotlib.pyplot as plt

    ncols = min(num_heads, 4)
    nrows = (num_heads + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.8 * nrows),
                             squeeze=False, sharex=True, sharey=True)

    layers = np.arange(num_layers)
    for h in range(num_heads):
        ax = axes[h // ncols, h % ncols]
        v_counts = np.zeros((len(samples_data), num_layers))
        s_counts = np.zeros((len(samples_data), num_layers))
        for si, data in enumerate(samples_data):
            for li in range(num_layers):
                v_counts[si, li] = (data[li]["v_idx"][h] != V_IDX_PAD).sum().item()
                s_counts[si, li] = (data[li]["s_idx"][h] != S_IDX_PAD).sum().item()

        vm, vs = v_counts.mean(axis=0), v_counts.std(axis=0)
        sm, ss = s_counts.mean(axis=0), s_counts.std(axis=0)

        ax.plot(layers, vm, color="steelblue", linewidth=1.2, label="V")
        ax.fill_between(layers, vm - vs, vm + vs, alpha=0.15, color="steelblue")
        ax.plot(layers, sm, color="coral", linewidth=1.2, label="S")
        ax.fill_between(layers, sm - ss, sm + ss, alpha=0.15, color="coral")
        ax.set_title(f"Head {h}", fontsize=9)
        ax.grid(True, alpha=0.3)
        if h == 0:
            ax.legend(fontsize="x-small")

    # Hide unused subplots
    for idx in range(num_heads, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.supxlabel("Layer")
    fig.supylabel("Valid index count")
    fig.suptitle(f"Per-head index counts (mean ± std over {len(samples_data)} samples)",
                 fontsize=13)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.95])
    out = args.output or (
        os.path.join(args.output_dir, "per_head_grid.png") if args.output_dir else None
    )
    _save_or_show(fig, out)


# ---------------------------------------------------------------------------
# Discreteness helpers
# ---------------------------------------------------------------------------


def _get_valid(tensor: np.ndarray, pad_value: int) -> np.ndarray:
    """Return valid (non-padding) entries from a 1-D array, sorted ascending.

    In the collected data, ``v_idx`` uses ``2147483647`` (INT32_MAX) as padding
    and ``s_idx`` uses ``-1``.  This helper strips those sentinels and returns
    the remaining values in ascending order, ready for gap analysis.
    """
    valid = tensor[tensor != pad_value]
    return np.sort(valid)


def _gap_stats(sorted_vals: np.ndarray) -> dict:
    """Compute discreteness statistics for a sorted 1-D array of index positions.

    Given a sorted sequence of selected token positions (e.g. vertical indices
    for one attention head at one layer), this function measures how
    "discrete" (scattered) vs. "contiguous" (clustered) the selection is.

    Definitions
    -----------
    **Gap**: the difference between two consecutive selected positions.
        ``gaps = sorted_vals[1:] - sorted_vals[:-1]``
        A gap of 1 means the two positions are adjacent (contiguous).
        A gap >> 1 means there is a large unselected stretch between them.

    **Run**: a maximal contiguous block of selected positions — i.e. a stretch
        where every consecutive pair has gap == 1.
        Example: positions [3, 4, 5, 10, 11, 20] contain 3 runs:
            run 1: [3, 4, 5]  (length 3)
            run 2: [10, 11]   (length 2)
            run 3: [20]       (length 1)
        Runs are identified by splitting at positions where gap > 1.

    **Fragmentation index**: a normalised measure of scatteredness.
        ``frag = (num_runs - 1) / (count - 1)``
        - **0.0**: all selected positions form a single contiguous block
          (one run covering all ``count`` indices).
        - **1.0**: every selected position is isolated (each run has length 1,
          so ``num_runs == count``).
        - Values in between indicate a mixture: some clustering, some gaps.
        This metric is independent of the total number of selected indices,
        making it comparable across heads/layers with different sparsity.

    Returned dict
    -------------
    count        : int   — number of valid (non-padding) positions
    num_gaps     : int   — ``count - 1`` (number of inter-position gaps)
    gap_mean     : float — average gap size (larger = more spread out)
    gap_median   : float — median gap (robust to outlier large gaps)
    gap_std      : float — gap variability (high = mix of tight clusters + far jumps)
    gap_min      : int   — smallest gap (typically 1 if any clustering exists)
    gap_max      : int   — largest gap (the widest "desert" between selected positions)
    num_runs     : int   — number of contiguous runs
    mean_run_len : float — average run length (larger = denser clusters)
    max_run_len  : int   — longest contiguous run
    frag_index   : float — fragmentation index in [0, 1] (see above)

    Interpretation guide
    --------------------
    - High ``frag_index`` + low ``mean_run_len`` (≈1):
        Indices are highly discrete / scattered — each selected token stands
        alone with no contiguous neighbours.  The attention pattern is a
        sparse "point cloud" over the sequence.

    - Low ``frag_index`` + high ``mean_run_len``:
        Indices cluster into a few large contiguous blocks — the attention
        pattern resembles a small number of dense windows.

    - High ``gap_std`` relative to ``gap_mean``:
        Gaps are heterogeneous — some regions are tightly packed while others
        have wide deserts.  Indicates structured rather than uniform spacing.

    - ``gap_median`` much smaller than ``gap_mean``:
        The gap distribution is right-skewed — most gaps are small (clusters)
        but a few very large gaps pull up the mean.
    """
    n = len(sorted_vals)
    if n <= 1:
        return {
            "count": n, "num_gaps": 0,
            "gap_mean": 0.0, "gap_median": 0.0, "gap_std": 0.0,
            "gap_min": 0, "gap_max": 0,
            "num_runs": n, "mean_run_len": float(n), "max_run_len": n,
            "frag_index": 0.0,
        }
    gaps = np.diff(sorted_vals)
    # Runs: contiguous stretches where gap == 1
    run_boundaries = np.where(gaps > 1)[0]
    num_runs = len(run_boundaries) + 1
    run_starts = np.concatenate([[0], run_boundaries + 1])
    run_ends = np.concatenate([run_boundaries + 1, [n]])
    run_lengths = run_ends - run_starts

    return {
        "count": n,
        "num_gaps": len(gaps),
        "gap_mean": float(np.mean(gaps)),
        "gap_median": float(np.median(gaps)),
        "gap_std": float(np.std(gaps)),
        "gap_min": int(np.min(gaps)),
        "gap_max": int(np.max(gaps)),
        "num_runs": int(num_runs),
        "mean_run_len": float(np.mean(run_lengths)),
        "max_run_len": int(np.max(run_lengths)),
        "frag_index": float((num_runs - 1) / max(n - 1, 1)),
    }


# ---------------------------------------------------------------------------
# Analysis: discreteness  (quantitative summary)
# ---------------------------------------------------------------------------


def mode_discreteness(samples_data, sample_indices, num_layers, num_heads, args):
    """Print a per-layer table of discreteness metrics (quantitative summary).

    This is the primary quantitative tool for answering "how discrete are the
    selected sparse attention indices?"  It computes gap and run statistics
    (see ``_gap_stats`` for metric definitions) for every (sample, head) pair
    and reports the *mean* across all samples × heads for each layer.

    Table columns
    -------------
    Layer   : transformer layer index
    Count   : mean number of valid (non-padding) indices per head
    Runs    : mean number of contiguous runs (blocks of adjacent positions)
    Frag    : mean fragmentation index (0 = one solid block, 1 = all isolated)
    GapMean : mean of inter-index gap sizes
    GapMed  : median gap (robust to outlier-large deserts)
    GapStd  : std dev of gap sizes (high = heterogeneous spacing)
    GapMin  : mean of per-head minimum gap (typically 1 if any clustering)
    GapMax  : mean of per-head maximum gap (largest desert between selections)
    RunMean : mean contiguous run length (higher = denser local clusters)
    RunMax  : mean of per-head longest run (largest contiguous block)

    How to read the output
    ----------------------
    - **Frag ≈ 0.0–0.2, RunMean > 10**: indices are concentrated in a few
      large contiguous blocks — the attention selects dense windows.
    - **Frag ≈ 0.8–1.0, RunMean ≈ 1**: indices are maximally scattered —
      nearly every selected position stands alone.
    - **GapMed << GapMean**: right-skewed gap distribution — most indices are
      close together (clusters) with occasional large gaps between clusters.
    - **GapStd / GapMean >> 1**: very heterogeneous — a mix of tight clusters
      and wide deserts.  Check ``gap_histogram`` mode for the full shape.

    Vertical vs. slash comparison: vertical indices select *token columns*
    (which keys are globally important), so they tend to be more scattered.
    Slash indices select *diagonals* (relative-position patterns), so they
    may show more structure (recency bias → clustering near recent tokens).
    """
    show_layers = _select_layers(num_layers, args.layers)

    # Accumulate stats across samples and heads
    for idx_type, pad, label in [
        ("v_idx", V_IDX_PAD, "VERTICAL"),
        ("s_idx", S_IDX_PAD, "SLASH"),
    ]:
        print(f"\n{'=' * 80}")
        print(f"  {label} INDEX DISCRETENESS")
        print(f"{'=' * 80}")
        print(
            f"{'Layer':<7} {'Count':>6} {'Runs':>6} {'Frag':>6} "
            f"{'GapMean':>8} {'GapMed':>8} {'GapStd':>8} "
            f"{'GapMin':>7} {'GapMax':>7} {'RunMean':>8} {'RunMax':>7}"
        )
        print("-" * 93)
        for li in show_layers:
            all_stats = []
            for data in samples_data:
                for h in range(num_heads):
                    vals = _get_valid(data[li][idx_type][h].numpy(), pad)
                    all_stats.append(_gap_stats(vals))
            # Average over samples × heads
            keys = [
                "count", "num_runs", "frag_index",
                "gap_mean", "gap_median", "gap_std", "gap_min", "gap_max",
                "mean_run_len", "max_run_len",
            ]
            avgs = {k: np.mean([s[k] for s in all_stats]) for k in keys}
            print(
                f"{li:<7} {avgs['count']:6.0f} {avgs['num_runs']:6.0f} "
                f"{avgs['frag_index']:6.3f} "
                f"{avgs['gap_mean']:8.1f} {avgs['gap_median']:8.1f} "
                f"{avgs['gap_std']:8.1f} "
                f"{avgs['gap_min']:7.0f} {avgs['gap_max']:7.0f} "
                f"{avgs['mean_run_len']:8.2f} {avgs['max_run_len']:7.0f}"
            )

    print()
    print("Frag(mentation) index: 0 = one contiguous block, 1 = all indices isolated")
    print("Runs: count of contiguous stretches (gap == 1 between neighbours)")


# ---------------------------------------------------------------------------
# Analysis: gap_histogram
# ---------------------------------------------------------------------------


def mode_gap_histogram(samples_data, sample_indices, num_layers, num_heads, args):
    """Histogram of inter-index gaps for selected layers (qualitative view).

    For each selected layer, plots the distribution of gap sizes (distance
    between consecutive selected positions) across all samples and heads.
    Uses **log-scale x-axis** so both small gaps (≈1, contiguous) and large
    gaps (100s–1000s, scattered) are visible in the same plot.

    Reading the plot
    ----------------
    - **Peak at gap=1**: many consecutive positions are selected together,
      forming contiguous runs.  The taller this peak relative to the rest,
      the more clustered the pattern.
    - **Flat / wide distribution**: gaps are spread across many scales —
      indices are scattered without a dominant spacing pattern.
    - **Bimodal** (peak at 1 + peak at large values): a mix of tight local
      clusters separated by large deserts.  This is the "structured discrete"
      pattern where attention selects a few dense windows plus isolated
      globally-important tokens.
    - **Median line** (dashed): if median << mean, the distribution is
      right-skewed (dominated by small gaps with a few huge outliers).

    The vertical (token-column) and slash (diagonal) gap distributions are
    shown side by side for comparison at each layer.
    """
    import matplotlib.pyplot as plt

    show_layers = _select_layers(num_layers, args.layers)
    fig, axes = plt.subplots(len(show_layers), 2, figsize=(14, 2.8 * len(show_layers)),
                             squeeze=False)

    for row, li in enumerate(show_layers):
        for col, (idx_type, pad, color, label) in enumerate([
            ("v_idx", V_IDX_PAD, "steelblue", "Vertical"),
            ("s_idx", S_IDX_PAD, "coral", "Slash"),
        ]):
            all_gaps = []
            for data in samples_data:
                for h in range(num_heads):
                    vals = _get_valid(data[li][idx_type][h].numpy(), pad)
                    if len(vals) > 1:
                        all_gaps.append(np.diff(vals))
            ax = axes[row, col]
            if all_gaps:
                gaps = np.concatenate(all_gaps)
                # Log-scale bins to capture both small and large gaps
                if gaps.max() > gaps.min() and gaps.min() >= 1:
                    bins = np.logspace(
                        np.log10(max(1, gaps.min())),
                        np.log10(gaps.max()),
                        60,
                    )
                    ax.hist(gaps, bins=bins, alpha=0.7, color=color, edgecolor="none")
                    ax.set_xscale("log")
                else:
                    ax.hist(gaps, bins=60, alpha=0.7, color=color, edgecolor="none")
                med = np.median(gaps)
                ax.axvline(med, color="black", linestyle="--", linewidth=1,
                           label=f"median={med:.0f}")
                ax.legend(fontsize="small")
            ax.set_title(f"Layer {li} — {label} gaps")
            ax.set_xlabel("Gap size (log scale)")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)

    fig.suptitle("Inter-index gap distribution (all samples × heads)", fontsize=14, y=1.01)
    fig.tight_layout()
    out = args.output or (
        os.path.join(args.output_dir, "gap_histogram.png") if args.output_dir else None
    )
    _save_or_show(fig, out)


# ---------------------------------------------------------------------------
# Analysis: fragmentation  (fragmentation index across layers)
# ---------------------------------------------------------------------------


def mode_fragmentation(samples_data, sample_indices, num_layers, num_heads, args):
    """Plot fragmentation index and mean run length across all layers.

    Two-panel line plot showing how discreteness evolves through the network,
    with ±1 standard-deviation bands computed over samples × heads.

    Panel 1 — Fragmentation index (left)
    -------------------------------------
    Y-axis: fragmentation index in [0, 1].
        frag = (num_runs − 1) / (count − 1)

    - Near 0: the selected indices form one or very few contiguous blocks.
      Attention focuses on compact windows.
    - Near 1: every selected index is isolated (run length ≈ 1).
      Attention picks scattered individual tokens.
    - Trend across layers: if fragmentation rises in deeper layers, deeper
      attention heads select sparser, more discrete patterns.  If it drops,
      deeper layers prefer denser, more local windows.

    Panel 2 — Mean run length (right)
    ----------------------------------
    Y-axis: average length of contiguous runs.

    - A mean run length of 1 means essentially no consecutive positions are
      selected together — maximally discrete.
    - A mean run length of 50 means selected positions typically come in
      blocks of ~50 adjacent tokens — strongly clustered.
    - The ±1σ band shows head-to-head variability: a wide band means some
      heads are clustered while others are scattered at the same layer.

    Vertical vs. slash lines are overlaid for direct comparison.
    """
    import matplotlib.pyplot as plt

    v_frag = np.zeros((len(samples_data), num_layers, num_heads))
    s_frag = np.zeros((len(samples_data), num_layers, num_heads))
    v_runlen = np.zeros_like(v_frag)
    s_runlen = np.zeros_like(s_frag)

    for si, data in enumerate(samples_data):
        for li in range(num_layers):
            for h in range(num_heads):
                vs = _gap_stats(_get_valid(data[li]["v_idx"][h].numpy(), V_IDX_PAD))
                ss = _gap_stats(_get_valid(data[li]["s_idx"][h].numpy(), S_IDX_PAD))
                v_frag[si, li, h] = vs["frag_index"]
                s_frag[si, li, h] = ss["frag_index"]
                v_runlen[si, li, h] = vs["mean_run_len"]
                s_runlen[si, li, h] = ss["mean_run_len"]

    layers = np.arange(num_layers)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Fragmentation index
    ax = axes[0]
    for arr, color, label in [
        (v_frag, "steelblue", "Vertical"),
        (s_frag, "coral", "Slash"),
    ]:
        mean = arr.mean(axis=(0, 2))  # mean over samples & heads
        std = arr.std(axis=(0, 2))
        ax.plot(layers, mean, color=color, linewidth=2, label=label)
        ax.fill_between(layers, mean - std, mean + std, alpha=0.15, color=color)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fragmentation index")
    ax.set_title("Fragmentation (0=contiguous, 1=fully scattered)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, num_layers - 1)
    ax.set_ylim(-0.02, 1.05)

    # Panel 2: Mean run length
    ax = axes[1]
    for arr, color, label in [
        (v_runlen, "steelblue", "Vertical"),
        (s_runlen, "coral", "Slash"),
    ]:
        mean = arr.mean(axis=(0, 2))
        std = arr.std(axis=(0, 2))
        ax.plot(layers, mean, color=color, linewidth=2, label=label)
        ax.fill_between(layers, mean - std, mean + std, alpha=0.15, color=color)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean run length")
    ax.set_title("Mean contiguous run length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, num_layers - 1)

    fig.suptitle(
        f"Index discreteness across layers (mean ± std over {len(samples_data)} samples × {num_heads} heads)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = args.output or (
        os.path.join(args.output_dir, "fragmentation.png") if args.output_dir else None
    )
    _save_or_show(fig, out)


# ---------------------------------------------------------------------------
# Analysis: raster  (spike raster of selected positions)
# ---------------------------------------------------------------------------


def mode_raster(samples_data, sample_indices, num_layers, num_heads, args):
    """Raster (spike) plot of selected index positions (qualitative view).

    Produces a plot inspired by neural spike rasters: each row is an
    attention head, and each dot marks a selected position along the
    sequence axis.  This gives an immediate *qualitative* impression of
    discreteness that complements the quantitative metrics:

    Visual patterns
    ---------------
    - **Dense horizontal bands**: the head selects large contiguous blocks
      of the sequence — low fragmentation, high run length.
    - **Sparse scattered dots**: the head picks isolated token positions —
      high fragmentation, the pattern is very discrete.
    - **Vertical stripes across heads**: certain sequence positions are
      selected by *all* heads (globally important "sink" tokens, e.g. BOS
      or other anchor tokens).
    - **Diagonal structure** (slash panel): slash indices represent diagonal
      offsets in the QK attention matrix, so a diagonal trend in the raster
      reflects recency-biased selection.
    - **Head-to-head variation**: different rows may show very different
      patterns — some heads are discrete while others are clustered.

    Defaults to showing the first, middle, and last layers.  Use ``--layers``
    to select specific layers.  Shows the first sample only (sample 0).
    """
    import matplotlib.pyplot as plt

    if args.layers == "auto":
        show_layers = [0, num_layers // 2, num_layers - 1]
    else:
        show_layers = _select_layers(num_layers, args.layers)

    sample_idx = 0  # Use first sample for the raster

    fig, axes = plt.subplots(
        len(show_layers), 2,
        figsize=(16, 1.5 + 1.0 * num_heads * len(show_layers)),
        squeeze=False,
    )

    for row, li in enumerate(show_layers):
        data = samples_data[sample_idx]
        for col, (idx_type, pad, color, label) in enumerate([
            ("v_idx", V_IDX_PAD, "steelblue", "Vertical"),
            ("s_idx", S_IDX_PAD, "coral", "Slash"),
        ]):
            ax = axes[row, col]
            for h in range(num_heads):
                vals = _get_valid(data[li][idx_type][h].numpy(), pad)
                ax.scatter(
                    vals, np.full_like(vals, h), s=0.3, color=color,
                    alpha=0.6, linewidths=0, rasterized=True,
                )
            ax.set_yticks(range(num_heads))
            ax.set_yticklabels([str(h) for h in range(num_heads)], fontsize=7)
            ax.set_ylabel("Head")
            ax.set_xlabel("Sequence position")
            ax.set_title(f"Layer {li} — {label}")
            ax.set_ylim(-0.5, num_heads - 0.5)
            ax.invert_yaxis()
            ax.grid(True, alpha=0.2, axis="x")

    fig.suptitle(
        f"Index raster plot (sample {sample_indices[sample_idx]})", fontsize=13, y=1.01
    )
    fig.tight_layout()
    out = args.output or (
        os.path.join(args.output_dir, "raster.png") if args.output_dir else None
    )
    _save_or_show(fig, out)


# ---------------------------------------------------------------------------
# Analysis: density_heatmap  (binned spatial density)
# ---------------------------------------------------------------------------


def mode_density_heatmap(samples_data, sample_indices, num_layers, num_heads, args):
    """Heatmap of index density across sequence regions (qualitative view).

    Divides the full sequence into ``num_bins`` (default 64) equal-width
    spatial bins and counts what fraction of selected indices fall into each
    bin.  The result is a 2-D heatmap: rows = layers, columns = sequence
    regions.

    Reading the plot
    ----------------
    - **Uniform colour across columns**: indices are spread roughly evenly
      over the sequence — low spatial discreteness (positions are not
      clustered in particular regions).
    - **Bright hot-spots / dark cold-spots**: indices concentrate in specific
      sequence regions.  This is *spatial* discreteness — not just that
      individual indices are isolated, but that the *density* varies across
      the sequence.
    - **Hot columns at position 0**: typical "sink token" behaviour — the
      BOS / beginning tokens are always selected (high vertical importance).
    - **Hot columns near the end**: recency effect — recent tokens are
      favoured, especially by slash (diagonal) patterns.
    - **Layer trends** (top-to-bottom patterns): if hot-spots shift position
      across layers, different layers attend to different parts of the
      context.  If the pattern is constant, the model has a fixed spatial
      bias regardless of depth.

    Vertical and slash density are shown side by side.  The colour scale is
    normalised *per layer* (each row sums to 1.0), so the heatmap shows
    *where* indices fall, not *how many* there are.
    """
    import matplotlib.pyplot as plt

    num_bins = 64

    # Infer sequence length
    seq_len = 0
    for data in samples_data:
        for li in range(num_layers):
            v = data[li]["v_idx"].numpy()
            valid = v[v != V_IDX_PAD]
            if len(valid) > 0:
                seq_len = max(seq_len, int(valid.max()) + 1)
    if seq_len == 0:
        print("Cannot infer sequence length.")
        return

    bin_edges = np.linspace(0, seq_len, num_bins + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, num_layers * 0.22)))

    for ax, (idx_type, pad, label, cmap) in zip(axes, [
        ("v_idx", V_IDX_PAD, "Vertical", "Blues"),
        ("s_idx", S_IDX_PAD, "Slash", "Oranges"),
    ]):
        # density[layer, bin] = mean fraction of indices in that bin
        density = np.zeros((num_layers, num_bins))
        for li in range(num_layers):
            bin_counts = np.zeros(num_bins)
            total_valid = 0
            for data in samples_data:
                for h in range(num_heads):
                    vals = _get_valid(data[li][idx_type][h].numpy(), pad)
                    if len(vals) > 0:
                        hist, _ = np.histogram(vals, bins=bin_edges)
                        bin_counts += hist
                        total_valid += len(vals)
            if total_valid > 0:
                density[li] = bin_counts / total_valid

        im = ax.imshow(
            density, aspect="auto", cmap=cmap, interpolation="nearest",
            extent=[0, seq_len, num_layers - 0.5, -0.5],
        )
        ax.set_xlabel("Sequence position")
        ax.set_ylabel("Layer")
        ax.set_title(f"{label} index density")
        fig.colorbar(im, ax=ax, shrink=0.6, label="Fraction of indices")

    fig.suptitle(
        f"Spatial density of selected indices ({len(samples_data)} samples × {num_heads} heads)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = args.output or (
        os.path.join(args.output_dir, "density_heatmap.png") if args.output_dir else None
    )
    _save_or_show(fig, out)


# ===========================================================================
# CROSS-ITERATION (DYNAMICS) ANALYSIS MODES
#
# These modes require --base_dir (not --data_dir) and load data from
# multiple checkpoint iterations to show how patterns evolve during training.
# ===========================================================================


def _require_multi(args):
    """Ensure multi-iteration data is available on args."""
    if not hasattr(args, "_multi_data") or args._multi_data is None:
        raise RuntimeError(
            "This mode requires --base_dir (multi-iteration data).  "
            "Use --base_dir pointing to a directory with 0000-XXXX/ subdirs."
        )
    return args._multi_data, args._ckpt_tags


# ---------------------------------------------------------------------------
# Dynamics: frag_over_steps
# ---------------------------------------------------------------------------


def mode_frag_over_steps(samples_data, sample_indices, num_layers, num_heads, args):
    """Fragmentation index vs. training step, one line per layer.

    Shows how the discreteness of selected indices evolves over the course
    of training.  Rising lines = attention becomes more scattered over
    training.  Falling lines = attention consolidates into denser blocks.
    """
    import matplotlib.pyplot as plt

    multi_data, ckpt_tags = _require_multi(args)
    show_layers = _select_layers(num_layers, args.layers)
    steps = [ckpt_tag_to_step(t) for t in ckpt_tags]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (idx_type, pad, title) in zip(axes, [
        ("v_idx", V_IDX_PAD, "Vertical fragmentation"),
        ("s_idx", S_IDX_PAD, "Slash fragmentation"),
    ]):
        cmap = plt.cm.viridis(np.linspace(0, 1, len(show_layers)))
        for ci, li in enumerate(show_layers):
            frag_per_step = []
            for tag in ckpt_tags:
                frags = []
                for data in multi_data[tag]:
                    for h in range(num_heads):
                        vals = _get_valid(data[li][idx_type][h].numpy(), pad)
                        frags.append(_gap_stats(vals)["frag_index"])
                frag_per_step.append(np.mean(frags))
            ax.plot(steps, frag_per_step, color=cmap[ci], alpha=0.8,
                    label=f"L{li}", linewidth=1.5)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Fragmentation index")
        ax.set_title(title)
        ax.legend(fontsize="x-small", ncol=max(1, len(show_layers) // 6))
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.02, 1.05)

    fig.suptitle("Fragmentation dynamics across training", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = args.output or (
        os.path.join(args.output_dir, "frag_over_steps.png") if args.output_dir else None
    )
    _save_or_show(fig, out)


# ---------------------------------------------------------------------------
# Dynamics: count_over_steps
# ---------------------------------------------------------------------------


def mode_count_over_steps(samples_data, sample_indices, num_layers, num_heads, args):
    """Valid index count vs. training step, one line per layer.

    Shows whether the number of selected indices (effective sparsity)
    changes during training.  For flex patterns, the count can vary
    as the model learns; for fixed patterns, it stays constant.
    """
    import matplotlib.pyplot as plt

    multi_data, ckpt_tags = _require_multi(args)
    show_layers = _select_layers(num_layers, args.layers)
    steps = [ckpt_tag_to_step(t) for t in ckpt_tags]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (idx_type, pad, title) in zip(axes, [
        ("v_idx", V_IDX_PAD, "Vertical count"),
        ("s_idx", S_IDX_PAD, "Slash count"),
    ]):
        cmap = plt.cm.viridis(np.linspace(0, 1, len(show_layers)))
        for ci, li in enumerate(show_layers):
            count_per_step = []
            for tag in ckpt_tags:
                counts = []
                for data in multi_data[tag]:
                    for h in range(num_heads):
                        n = (data[li][idx_type][h] != pad).sum().item()
                        counts.append(n)
                count_per_step.append(np.mean(counts))
            ax.plot(steps, count_per_step, color=cmap[ci], alpha=0.8,
                    label=f"L{li}", linewidth=1.5)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Mean valid index count")
        ax.set_title(title)
        ax.legend(fontsize="x-small", ncol=max(1, len(show_layers) // 6))
        ax.grid(True, alpha=0.3)

    fig.suptitle("Index count dynamics across training", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = args.output or (
        os.path.join(args.output_dir, "count_over_steps.png") if args.output_dir else None
    )
    _save_or_show(fig, out)


# ---------------------------------------------------------------------------
# Dynamics: dynamics_heatmap
# ---------------------------------------------------------------------------


def mode_dynamics_heatmap(samples_data, sample_indices, num_layers, num_heads, args):
    """Heatmap: layers (y) × training steps (x), coloured by fragmentation.

    Provides a bird's-eye view of how discreteness varies across both layers
    and training time simultaneously.  Hot cells = highly scattered indices;
    cool cells = contiguous blocks.
    """
    import matplotlib.pyplot as plt

    multi_data, ckpt_tags = _require_multi(args)
    steps = [ckpt_tag_to_step(t) for t in ckpt_tags]

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(ckpt_tags) * 0.4),
                                            max(6, num_layers * 0.22)))
    for ax, (idx_type, pad, title, cmap_name) in zip(axes, [
        ("v_idx", V_IDX_PAD, "Vertical frag.", "YlOrRd"),
        ("s_idx", S_IDX_PAD, "Slash frag.", "YlOrRd"),
    ]):
        mat = np.zeros((num_layers, len(ckpt_tags)))
        for ci, tag in enumerate(ckpt_tags):
            for li in range(num_layers):
                frags = []
                for data in multi_data[tag]:
                    for h in range(num_heads):
                        vals = _get_valid(data[li][idx_type][h].numpy(), pad)
                        frags.append(_gap_stats(vals)["frag_index"])
                mat[li, ci] = np.mean(frags)

        im = ax.imshow(mat, aspect="auto", cmap=cmap_name, interpolation="nearest",
                       vmin=0, vmax=1)
        ax.set_xticks(range(len(ckpt_tags)))
        ax.set_xticklabels([str(s) for s in steps], rotation=90, fontsize=7)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Layer")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.6, label="Fragmentation")

    fig.suptitle("Fragmentation across layers and training steps", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = args.output or (
        os.path.join(args.output_dir, "dynamics_heatmap.png") if args.output_dir else None
    )
    _save_or_show(fig, out)


# ---------------------------------------------------------------------------
# Dynamics: overlap_over_steps
# ---------------------------------------------------------------------------


def mode_overlap_over_steps(samples_data, sample_indices, num_layers, num_heads, args):
    """Jaccard similarity between consecutive checkpoints vs. training step.

    Measures how much the *set* of selected indices changes from one
    training iteration to the next.  High similarity = stable selections;
    low similarity = the model keeps picking different tokens.

    For each pair of consecutive checkpoints, computes per-head Jaccard
    between the index sets (using the first sample), then averages over
    heads and selected layers.
    """
    import matplotlib.pyplot as plt

    multi_data, ckpt_tags = _require_multi(args)
    if len(ckpt_tags) < 2:
        print("Need at least 2 checkpoints for overlap_over_steps.")
        return

    show_layers = _select_layers(num_layers, args.layers)
    steps = [ckpt_tag_to_step(t) for t in ckpt_tags]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (idx_type, pad, title) in zip(axes, [
        ("v_idx", V_IDX_PAD, "Vertical stability"),
        ("s_idx", S_IDX_PAD, "Slash stability"),
    ]):
        cmap = plt.cm.viridis(np.linspace(0, 1, len(show_layers)))
        for ci, li in enumerate(show_layers):
            jac_per_pair = []
            for ti in range(len(ckpt_tags) - 1):
                jacs = []
                data_a = multi_data[ckpt_tags[ti]][0]     # first sample
                data_b = multi_data[ckpt_tags[ti + 1]][0]
                for h in range(num_heads):
                    va = _get_valid(data_a[li][idx_type][h].numpy(), pad)
                    vb = _get_valid(data_b[li][idx_type][h].numpy(), pad)
                    jacs.append(_jaccard(va, vb))
                jac_per_pair.append(np.mean(jacs))
            ax.plot(steps[1:], jac_per_pair, color=cmap[ci], alpha=0.8,
                    label=f"L{li}", linewidth=1.5)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Jaccard similarity (vs. previous step)")
        ax.set_title(title)
        ax.legend(fontsize="x-small", ncol=max(1, len(show_layers) // 6))
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.02, 1.05)

    fig.suptitle("Index stability across consecutive training steps", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = args.output or (
        os.path.join(args.output_dir, "overlap_over_steps.png") if args.output_dir else None
    )
    _save_or_show(fig, out)


# ---------------------------------------------------------------------------
# Analysis: export_csv
# ---------------------------------------------------------------------------


def mode_export_csv(samples_data, sample_indices, num_layers, num_heads, args):
    """Export a flat CSV with one row per (sample, layer, head)."""
    output = args.output or os.path.join(args.output_dir or ".", "sparse_indices.csv")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    with open(output, "w") as f:
        f.write("sample,layer,head,v_count,s_count,v_indices,s_indices\n")
        for si_idx, si in enumerate(sample_indices):
            data = samples_data[si_idx]
            for li in range(num_layers):
                for h in range(num_heads):
                    v = data[li]["v_idx"][h].numpy()
                    v_valid = v[v != V_IDX_PAD]
                    s = data[li]["s_idx"][h].numpy()
                    s_valid = s[s != S_IDX_PAD]
                    v_str = " ".join(str(x) for x in v_valid)
                    s_str = " ".join(str(x) for x in s_valid)
                    f.write(
                        f"{si},{li},{h},{len(v_valid)},{len(s_valid)},"
                        f'"{v_str}","{s_str}"\n'
                    )
    print(f"Exported {output}")


# ---------------------------------------------------------------------------
# Analysis: draw_all
# ---------------------------------------------------------------------------


def mode_draw_all(samples_data, sample_indices, num_layers, num_heads, args):
    """Run all visual analysis modes and save PNGs to output_dir."""
    if not args.output_dir:
        args.output_dir = os.path.join(args.data_dir, "analysis")
    os.makedirs(args.output_dir, exist_ok=True)

    for mode_name, fn in MODES.items():
        if mode_name in ("draw_all", "export_csv", "summary"):
            continue
        print(f"--- {mode_name} ---")
        try:
            fn(samples_data, sample_indices, num_layers, num_heads, args)
        except Exception as e:
            print(f"  WARN: {mode_name} failed: {e}")

    # Also run summary (text)
    print("\n--- summary ---")
    mode_summary(samples_data, sample_indices, num_layers, num_heads, args)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

MODES = {
    # --- Single-iteration modes (--data_dir) ---
    "summary": mode_summary,
    "count_heatmap": mode_count_heatmap,
    "position_dist": mode_position_dist,
    "overlap": mode_overlap,
    "sparsity": mode_sparsity,
    "head_variance": mode_head_variance,
    "per_head_grid": mode_per_head_grid,
    "discreteness": mode_discreteness,
    "gap_histogram": mode_gap_histogram,
    "fragmentation": mode_fragmentation,
    "raster": mode_raster,
    "density_heatmap": mode_density_heatmap,
    "export_csv": mode_export_csv,
    "draw_all": mode_draw_all,
    # --- Cross-iteration dynamics modes (--base_dir) ---
    "frag_over_steps": mode_frag_over_steps,
    "count_over_steps": mode_count_over_steps,
    "dynamics_heatmap": mode_dynamics_heatmap,
    "overlap_over_steps": mode_overlap_over_steps,
}

# Modes that require multi-iteration data via --base_dir
_MULTI_ITER_MODES = {
    "frag_over_steps", "count_over_steps",
    "dynamics_heatmap", "overlap_over_steps",
}


def main():
    parser = argparse.ArgumentParser(
        description="Analyze sparse attention indices from single-card inference"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--data_dir", type=str, default=None,
        help="Directory containing sample_*.pt files (single iteration)",
    )
    group.add_argument(
        "--base_dir", type=str, default=None,
        help="Root directory with checkpoint-tag subdirs (e.g. 0000-0001/, "
             "0000-0002/, ...) for cross-iteration dynamics modes",
    )
    parser.add_argument(
        "--mode", type=str, default="summary", choices=list(MODES.keys()),
        help="Analysis mode (default: summary)",
    )
    parser.add_argument(
        "--layers", type=str, default="auto",
        help="Layer selection: 'all', 'auto', '0,5,10', '0-35:5'",
    )
    parser.add_argument(
        "--heads", type=str, default="all",
        help="Head selection: 'all', 'auto', '0,4,8'",
    )
    parser.add_argument(
        "--samples", type=str, default="all",
        help="Sample selection: 'all' or '0,1,2'",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit samples per checkpoint when using --base_dir (for speed)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (for single-file modes like export_csv or a plot)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for draw_all mode",
    )
    args = parser.parse_args()

    is_multi = args.mode in _MULTI_ITER_MODES

    if is_multi and args.base_dir is None:
        parser.error(f"Mode '{args.mode}' requires --base_dir (not --data_dir).")
    if not is_multi and args.data_dir is None and args.base_dir is not None:
        # When --base_dir is given for a single-iter mode, use the first
        # checkpoint subdir as data_dir so the mode still works.
        tags = discover_checkpoints(args.base_dir)
        if not tags:
            parser.error(f"No checkpoint subdirs found in {args.base_dir}")
        args.data_dir = os.path.join(args.base_dir, tags[-1])
        print(f"Using latest checkpoint: {tags[-1]}")

    # Load single-iteration data (always needed — used as the primary args
    # for mode functions; for multi-iter modes the first checkpoint is used)
    if args.samples != "all":
        sample_indices = [int(x) for x in args.samples.split(",")]
    else:
        sample_indices = None

    samples_data, sample_indices, num_layers, num_heads = load_all(
        args.data_dir, sample_indices
    )
    print(
        f"Loaded {len(samples_data)} samples, {num_layers} layers, {num_heads} heads "
        f"from {args.data_dir}"
    )

    # Load multi-iteration data if needed
    args._multi_data = None
    args._ckpt_tags = None
    if is_multi or (args.base_dir is not None):
        multi_data, ckpt_tags, _, _ = load_multi_iter(
            args.base_dir, max_samples=args.max_samples
        )
        args._multi_data = multi_data
        args._ckpt_tags = ckpt_tags
        print(f"Loaded {len(ckpt_tags)} checkpoints from {args.base_dir}")

    MODES[args.mode](samples_data, sample_indices, num_layers, num_heads, args)


if __name__ == "__main__":
    main()
