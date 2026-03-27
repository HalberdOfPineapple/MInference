#!/usr/bin/env python3
# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Unit tests for ``compute_sparse_ratio_local``.

Each test constructs a ``block_mask`` and ``bar_cnt`` with known values,
computes the expected sparse ratio by hand, and checks the function output.

Background
----------
The sparse attention decomposes the causal attention matrix into:

1. **Block attention** — dense computation on selected G×G blocks.
   ``block_mask[b, h, q_block, k_block] = True`` enables that block.
   In causal attention the mask is always lower-triangular (q ≥ k), and
   diagonal blocks (q == k) are always present but only ~half their
   entries are used due to causal masking within the block.

2. **Bar attention** — fine-grained attention to individual key positions
   outside the dense blocks (vertical + slash indices).
   ``bar_cnt[b, h, q_block, -1]`` counts how many such keys query block
   ``q_block`` attends to.

The formula (matching the distributed ``compute_sparse_ratio``)::

    active_block = block_mask.sum() * G²
    causal_corr  = nB * G² * B * H / 2        # diagonal blocks are half-used
    active_bar   = bar_cnt[..., -1].sum() * G  # each bar key covers G queries
    active       = (active_block - causal_corr) + active_bar
    total        = N² * B * H / 2              # full causal triangle
    sparse_ratio = 1 - active / total

The causal correction is a **fixed constant** (not dependent on which blocks
are active) because the block attention kernel *always* applies causal
masking on diagonal blocks.  This means the formula is only physically
meaningful when the mask follows causal structure: diagonal blocks are
always active, and only lower-triangular blocks may be turned on.

Run::

    python tests/test_sparse_ratio_local.py
"""

import sys

import torch

sys.path.insert(0, ".")
from minference.dist_ops.index_collector import compute_sparse_ratio_local


# ── helpers ─────────────────────────────────────────────────────────────────


def _causal_mask(B: int, H: int, nB: int) -> torch.Tensor:
    """Full causal (lower-triangular) block mask: [B, H, nB, nB]."""
    mask = torch.zeros(B, H, nB, nB, dtype=torch.bool)
    for q in range(nB):
        for k in range(q + 1):
            mask[:, :, q, k] = True
    return mask


def _diagonal_mask(B: int, H: int, nB: int) -> torch.Tensor:
    """Only diagonal blocks active (maximally sparse causal): [B, H, nB, nB]."""
    mask = torch.zeros(B, H, nB, nB, dtype=torch.bool)
    for i in range(nB):
        mask[:, :, i, i] = True
    return mask


def _expected_ratio(
    num_active_blocks: int,
    total_bar_entries: int,
    B: int, H: int, nB: int, G: int,
) -> float:
    """Compute the expected sparse ratio by hand."""
    N = nB * G
    block_area = G * G
    active_block = num_active_blocks * block_area
    causal_corr = nB * block_area * B * H / 2.0
    active_bar = total_bar_entries * G
    active = (active_block - causal_corr) + active_bar
    total = N * N * B * H / 2.0
    return 1.0 - active / total


def _run(name, block_mask, bar_cnt, num_tokens, granularity, expected, tol=1e-9):
    got = compute_sparse_ratio_local(block_mask, bar_cnt, num_tokens, granularity)
    ok = abs(got - expected) < tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: got={got:.10f}, expected={expected:.10f}")
    if not ok:
        raise AssertionError(f"{name}: {got} != {expected}")


# ── tests ───────────────────────────────────────────────────────────────────


def test_full_causal_is_zero():
    """Full causal lower-triangle → sparse ratio = 0 (dense attention).

    Active blocks = nB*(nB+1)/2 (lower triangle including diagonal).
    active_block = nB*(nB+1)/2 * G²
    causal_corr  = nB * G² / 2
    net          = nB*(nB+1)/2*G² - nB*G²/2 = nB²*G²/2
    total        = (nB*G)²/2 = nB²*G²/2
    ratio        = 0
    """
    B, H, nB, G = 1, 1, 8, 128
    N = nB * G
    mask = _causal_mask(B, H, nB)
    bar_cnt = torch.zeros(B, H, nB, 2, dtype=torch.int32)
    num_active = nB * (nB + 1) // 2
    expected = _expected_ratio(num_active, 0, B, H, nB, G)
    assert abs(expected) < 1e-12, f"expected 0, got {expected}"
    _run("full_causal_is_zero", mask, bar_cnt, N, G, expected)


def test_diagonal_only():
    """Only diagonal blocks → maximally sparse block pattern.

    active_block = nB * G²
    causal_corr  = nB * G² / 2
    net_block    = nB*G²/2
    total        = nB²*G²/2
    ratio        = 1 - 1/nB
    """
    B, H, nB, G = 1, 1, 8, 128
    N = nB * G
    mask = _diagonal_mask(B, H, nB)
    bar_cnt = torch.zeros(B, H, nB, 2, dtype=torch.int32)
    expected = _expected_ratio(nB, 0, B, H, nB, G)
    assert abs(expected - (1.0 - 1.0 / nB)) < 1e-12
    _run("diagonal_only", mask, bar_cnt, N, G, expected)


def test_diagonal_plus_bars():
    """Diagonal blocks + bar entries — typical sparse attention.

    nB=32, G=128 → N=4096.
    100 bar entries per query block → 3200 total.

    net_block = nB*G²/2 = 32*16384/2 = 262144
    active_bar = 3200*128 = 409600
    active = 671744
    total = 4096²/2 = 8388608
    ratio = 1 - 671744/8388608 ≈ 0.9199
    """
    B, H, nB, G = 1, 1, 32, 128
    N = nB * G
    mask = _diagonal_mask(B, H, nB)
    bar_per_block = 100
    bar_cnt = torch.zeros(B, H, nB, 2, dtype=torch.int32)
    bar_cnt[..., -1] = bar_per_block
    expected = _expected_ratio(nB, nB * bar_per_block, B, H, nB, G)
    assert 0 < expected < 1, f"expected in (0,1), got {expected}"
    _run("diagonal_plus_bars", mask, bar_cnt, N, G, expected)


def test_partial_causal():
    """Causal mask with some off-diagonal blocks removed → partial sparsity.

    nB=4.  Full causal has 4+3+2+1=10 active blocks.
    Remove 3 off-diagonal blocks → 7 active.
    """
    B, H, nB, G = 1, 1, 4, 64
    N = nB * G
    mask = _causal_mask(B, H, nB)
    # Remove blocks (1,0), (2,0), (3,0) — column 0 below row 0
    mask[0, 0, 1, 0] = False
    mask[0, 0, 2, 0] = False
    mask[0, 0, 3, 0] = False

    bar_cnt = torch.zeros(B, H, nB, 2, dtype=torch.int32)
    num_active = 10 - 3  # 7
    expected = _expected_ratio(num_active, 0, B, H, nB, G)
    assert 0 < expected < 1
    _run("partial_causal", mask, bar_cnt, N, G, expected)


def test_multi_batch_multi_head():
    """B=2, H=4 with full causal → ratio = 0, scaling is correct."""
    B, H, nB, G = 2, 4, 4, 64
    N = nB * G
    mask = _causal_mask(B, H, nB)
    bar_cnt = torch.zeros(B, H, nB, 2, dtype=torch.int32)
    num_active = B * H * nB * (nB + 1) // 2
    expected = _expected_ratio(num_active, 0, B, H, nB, G)
    assert abs(expected) < 1e-12
    _run("multi_batch_multi_head", mask, bar_cnt, N, G, expected)


def test_multi_head_varying_bars():
    """Different bar counts per head — sum across heads must be correct."""
    B, H, nB, G = 1, 2, 4, 64
    N = nB * G
    mask = _diagonal_mask(B, H, nB)
    bar_cnt = torch.zeros(B, H, nB, 2, dtype=torch.int32)
    # Head 0: bars = [10, 20, 30, 40], Head 1: bars = [5, 15, 25, 35]
    bar_cnt[0, 0, :, -1] = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
    bar_cnt[0, 1, :, -1] = torch.tensor([5, 15, 25, 35], dtype=torch.int32)
    total_bar = 10 + 20 + 30 + 40 + 5 + 15 + 25 + 35  # 180
    num_active = B * H * nB  # diagonal only
    expected = _expected_ratio(num_active, total_bar, B, H, nB, G)
    assert 0 < expected < 1
    _run("multi_head_varying_bars", mask, bar_cnt, N, G, expected)


def test_large_nB_high_sparsity():
    """Large sequence (nB=256, N=32768), diagonal + few bars → very sparse."""
    B, H, nB, G = 1, 1, 256, 128
    N = nB * G  # 32768
    mask = _diagonal_mask(B, H, nB)
    bar_per_block = 50
    bar_cnt = torch.zeros(B, H, nB, 2, dtype=torch.int32)
    bar_cnt[..., -1] = bar_per_block
    expected = _expected_ratio(nB, nB * bar_per_block, B, H, nB, G)
    # Diagonal-only ratio = 1 - 1/256 ≈ 0.996, bars add a tiny bit
    assert 0.98 < expected < 1.0, f"expected high sparsity, got {expected}"
    _run("large_nB_high_sparsity", mask, bar_cnt, N, G, expected)


def test_causal_plus_a_few_bars():
    """Full causal + a few bars → ratio slightly below 0.

    Full causal already covers the entire triangle (ratio=0).  Bar entries
    add extra computation on top, so ratio goes negative.  In practice
    bars overlap with blocks, but the formula counts them independently —
    this is consistent with the distributed version.
    """
    B, H, nB, G = 1, 1, 4, 64
    N = nB * G
    mask = _causal_mask(B, H, nB)
    bar_per_block = 20
    bar_cnt = torch.zeros(B, H, nB, 2, dtype=torch.int32)
    bar_cnt[..., -1] = bar_per_block
    num_active = nB * (nB + 1) // 2
    expected = _expected_ratio(num_active, nB * bar_per_block, B, H, nB, G)
    # Full causal = 0 + bars → slightly negative
    assert expected < 0
    _run("causal_plus_a_few_bars", mask, bar_cnt, N, G, expected)


def test_ratio_bounds_realistic():
    """Verify that for realistic sparse masks the ratio is in [0, 1).

    In practice, the number of bar entries is a small fraction of
    num_tokens — roughly the v_size + s_size from the pattern config.
    For flex_0.90, typical values are v_size=s_size≤4096 for sequences
    of 512K tokens.  We use a proportional bar count here.
    """
    for nB in [4, 8, 16, 32, 64]:
        B, H, G = 1, 1, 128
        N = nB * G
        mask = _diagonal_mask(B, H, nB)
        # Bar count proportional to sqrt(N) — well below N
        bar_per_block = max(1, int(N ** 0.5) // nB)
        bar_cnt = torch.zeros(B, H, nB, 2, dtype=torch.int32)
        bar_cnt[..., -1] = bar_per_block
        ratio = compute_sparse_ratio_local(mask, bar_cnt, N, G)
        assert 0 <= ratio <= 1, f"nB={nB}: ratio={ratio} out of [0,1]"
        print(f"  [PASS] nB={nB}, bar/block={bar_per_block}: ratio={ratio:.6f}")


# ── runner ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_full_causal_is_zero,
        test_diagonal_only,
        test_diagonal_plus_bars,
        test_partial_causal,
        test_multi_batch_multi_head,
        test_multi_head_varying_bars,
        test_large_nB_high_sparsity,
        test_causal_plus_a_few_bars,
        test_ratio_bounds_realistic,
    ]
    print(f"Running {len(tests)} tests for compute_sparse_ratio_local\n")
    passed = 0
    for fn in tests:
        print(f"{fn.__name__}:")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  ASSERTION ERROR: {e}")
    print(f"\n{passed}/{len(tests)} tests passed.")
    if passed < len(tests):
        sys.exit(1)
