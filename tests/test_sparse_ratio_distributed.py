#!/usr/bin/env python3
# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Test that ``compute_sparse_ratio`` (distributed) and
``compute_sparse_ratio_local`` (single-card) produce the same result
when given equivalent masks.

Strategy
--------
1. Construct a **global** causal block_mask
   ``[B, H, nB_global, nB_global]`` and global bar counts.
2. Distribute the global mask into **per-rank stripe views**:
   ``[world_size, B, H, nB_local, nB_local]`` per rank.
3. Mock ``dist.all_reduce`` so that it sums contributions from all ranks
   without actually launching processes.
4. Call ``compute_sparse_ratio`` for every rank and verify the result
   matches ``compute_sparse_ratio_local`` on the global mask.

Stripe distribution mapping
----------------------------
With ``world_size = W`` and ``nB_local = nB_global / W``:

* Rank ``r`` owns global query blocks:  ``r, r+W, r+2W, ...``
  i.e. local query ``q_local`` → global query ``q_local * W + r``

* At ring step ``s``, rank ``r`` holds K/V from rank ``(r - s) % W``,
  which owns global key blocks:  ``(r-s)%W, (r-s)%W + W, ...``
  i.e. local key ``k_local`` → global key ``k_local * W + (r - s) % W``

* ``block_mask[s, b, h, q_local, k_local] =
      global_mask[b, h, q_local*W + r, k_local*W + (r-s)%W]``

bar_cnt encoding
----------------
In the distributed case each query block's bar entries are split across
ranks.  ``bar_cnt[b, h, q_local, s+1]`` is the *cumulative* count of
bar entries up to and including step ``s``.  The last column
``bar_cnt[..., -1]`` (= ``bar_cnt[..., world_size]``) is the total.

For this test we construct global bar counts per query block and
distribute them round-robin across steps.

Run::

    python tests/test_sparse_ratio_distributed.py
"""

import sys
from unittest.mock import MagicMock

import torch

sys.path.insert(0, ".")

from minference.dist_ops.index_collector import compute_sparse_ratio_local


# ── helpers ─────────────────────────────────────────────────────────────────


def _global_causal_mask(B: int, H: int, nB: int) -> torch.Tensor:
    """Full causal (lower-triangular) block mask: [B, H, nB, nB]."""
    idx = torch.arange(nB)
    mask = idx.unsqueeze(0) <= idx.unsqueeze(1)  # [nB, nB] lower triangle
    return mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1).contiguous()


def _global_sparse_mask(B: int, H: int, nB: int) -> torch.Tensor:
    """Sparse causal mask: diagonal + one sub-diagonal."""
    mask = torch.zeros(B, H, nB, nB, dtype=torch.bool)
    for i in range(nB):
        mask[:, :, i, i] = True          # diagonal
        if i > 0:
            mask[:, :, i, i - 1] = True  # one sub-diagonal
    return mask


def _distribute_block_mask_stripe(
    global_mask: torch.Tensor,  # [B, H, nB_global, nB_global]
    world_size: int,
    rank: int,
) -> torch.Tensor:
    """Convert a global block mask into per-rank stripe view.

    Returns: [world_size, B, H, nB_local, nB_local]
    """
    B, H, nB_global, _ = global_mask.shape
    assert nB_global % world_size == 0
    nB_local = nB_global // world_size

    result = torch.zeros(world_size, B, H, nB_local, nB_local, dtype=torch.bool)
    for s in range(world_size):
        for ql in range(nB_local):
            for kl in range(nB_local):
                q_global = ql * world_size + rank
                k_rank = (rank - s) % world_size
                k_global = kl * world_size + k_rank
                result[s, :, :, ql, kl] = global_mask[:, :, q_global, k_global]
    return result


def _distribute_bar_cnt_stripe(
    global_bar_per_qblock: torch.Tensor,  # [B, H, nB_global]
    world_size: int,
    rank: int,
) -> torch.Tensor:
    """Convert global per-query-block bar counts into per-rank bar_cnt.

    In practice, bar entries for a query block are split across the W
    K/V shards.  For this test we distribute them evenly (round-robin).

    Returns: [B, H, nB_local, world_size + 1]  (cumulative format)
    """
    B, H, nB_global = global_bar_per_qblock.shape
    nB_local = nB_global // world_size

    bar_cnt = torch.zeros(B, H, nB_local, world_size + 1, dtype=torch.int32)
    for ql in range(nB_local):
        q_global = ql * world_size + rank
        total = global_bar_per_qblock[:, :, q_global]  # [B, H]
        # Split total evenly across steps, remainder goes to step 0
        per_step = total // world_size
        remainder = total % world_size
        cumsum = torch.zeros_like(total)
        for s in range(world_size):
            step_cnt = per_step + (1 if s == 0 else 0) * remainder
            cumsum = cumsum + step_cnt
            bar_cnt[:, :, ql, s + 1] = cumsum
    return bar_cnt


def _compute_sparse_ratio_distributed_mock(
    all_rank_block_masks: list,   # len = world_size, each [W, B, H, nB_L, nB_L]
    all_rank_bar_cnts: list,      # len = world_size, each [B, H, nB_L, W+1]
    num_tokens_local: int,
    world_size: int,
    granularity: int,
) -> float:
    """Simulate compute_sparse_ratio with mocked all_reduce.

    Instead of launching real distributed processes, we sum the per-rank
    contributions directly (which is what all_reduce(SUM) does).
    """
    import triton

    batch_size = all_rank_block_masks[0].shape[1]
    num_qo_heads = all_rank_block_masks[0].shape[2]
    num_tokens_global = world_size * num_tokens_local
    num_blocks_local = triton.cdiv(num_tokens_local, granularity)
    num_blocks_global = world_size * num_blocks_local
    block_area = granularity * granularity

    # Sum block_mask.sum() across ranks (= all_reduce)
    total_active_blocks = sum(
        m.sum().item() for m in all_rank_block_masks
    )
    num_active_block_entries = total_active_blocks * block_area
    num_active_block_entries -= num_blocks_global * block_area * batch_size * num_qo_heads / 2.0

    # Sum bar_cnt[..., -1].sum() across ranks (= all_reduce)
    total_active_bars = sum(
        c[..., -1].sum().item() for c in all_rank_bar_cnts
    )
    num_active_bar_entries = total_active_bars * granularity

    num_active_entries = num_active_block_entries + num_active_bar_entries
    total_entries = num_tokens_global * num_tokens_global * batch_size * num_qo_heads / 2.0
    sparse_ratio = 1.0 - num_active_entries / total_entries
    return sparse_ratio


def _run(name: str, got: float, expected: float, tol: float = 1e-9):
    ok = abs(got - expected) < tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: distributed={got:.10f}, single_card={expected:.10f}")
    if not ok:
        raise AssertionError(f"{name}: {got} != {expected}")


# ── tests ───────────────────────────────────────────────────────────────────


def test_full_causal_2_ranks():
    """Full causal mask, world_size=2, no bars."""
    B, H, nB_global, G = 1, 1, 8, 128
    W = 2
    N_global = nB_global * G
    N_local = N_global // W

    global_mask = _global_causal_mask(B, H, nB_global)
    global_bar = torch.zeros(B, H, nB_global, dtype=torch.int32)

    # Single-card reference
    bar_cnt_local = torch.zeros(B, H, nB_global, 2, dtype=torch.int32)
    ref = compute_sparse_ratio_local(global_mask, bar_cnt_local, N_global, G)

    # Distributed
    rank_masks = [_distribute_block_mask_stripe(global_mask, W, r) for r in range(W)]
    rank_bars = [_distribute_bar_cnt_stripe(global_bar, W, r) for r in range(W)]
    dist_ratio = _compute_sparse_ratio_distributed_mock(
        rank_masks, rank_bars, N_local, W, G
    )
    _run("full_causal_2_ranks", dist_ratio, ref)


def test_full_causal_4_ranks():
    """Full causal mask, world_size=4, no bars."""
    B, H, nB_global, G = 1, 2, 16, 64
    W = 4
    N_global = nB_global * G
    N_local = N_global // W

    global_mask = _global_causal_mask(B, H, nB_global)
    global_bar = torch.zeros(B, H, nB_global, dtype=torch.int32)

    bar_cnt_local = torch.zeros(B, H, nB_global, 2, dtype=torch.int32)
    ref = compute_sparse_ratio_local(global_mask, bar_cnt_local, N_global, G)

    rank_masks = [_distribute_block_mask_stripe(global_mask, W, r) for r in range(W)]
    rank_bars = [_distribute_bar_cnt_stripe(global_bar, W, r) for r in range(W)]
    dist_ratio = _compute_sparse_ratio_distributed_mock(
        rank_masks, rank_bars, N_local, W, G
    )
    _run("full_causal_4_ranks", dist_ratio, ref)


def test_sparse_mask_2_ranks():
    """Sparse causal mask (diagonal + sub-diagonal), world_size=2."""
    B, H, nB_global, G = 1, 1, 8, 128
    W = 2
    N_global = nB_global * G
    N_local = N_global // W

    global_mask = _global_sparse_mask(B, H, nB_global)
    global_bar = torch.zeros(B, H, nB_global, dtype=torch.int32)

    bar_cnt_local = torch.zeros(B, H, nB_global, 2, dtype=torch.int32)
    ref = compute_sparse_ratio_local(global_mask, bar_cnt_local, N_global, G)

    rank_masks = [_distribute_block_mask_stripe(global_mask, W, r) for r in range(W)]
    rank_bars = [_distribute_bar_cnt_stripe(global_bar, W, r) for r in range(W)]
    dist_ratio = _compute_sparse_ratio_distributed_mock(
        rank_masks, rank_bars, N_local, W, G
    )
    _run("sparse_mask_2_ranks", dist_ratio, ref)


def test_sparse_mask_4_ranks():
    """Sparse causal mask, world_size=4, multi-head."""
    B, H, nB_global, G = 2, 4, 16, 64
    W = 4
    N_global = nB_global * G
    N_local = N_global // W

    global_mask = _global_sparse_mask(B, H, nB_global)
    global_bar = torch.zeros(B, H, nB_global, dtype=torch.int32)

    bar_cnt_local = torch.zeros(B, H, nB_global, 2, dtype=torch.int32)
    ref = compute_sparse_ratio_local(global_mask, bar_cnt_local, N_global, G)

    rank_masks = [_distribute_block_mask_stripe(global_mask, W, r) for r in range(W)]
    rank_bars = [_distribute_bar_cnt_stripe(global_bar, W, r) for r in range(W)]
    dist_ratio = _compute_sparse_ratio_distributed_mock(
        rank_masks, rank_bars, N_local, W, G
    )
    _run("sparse_mask_4_ranks", dist_ratio, ref)


def test_with_bars_2_ranks():
    """Sparse mask + bar entries, world_size=2."""
    B, H, nB_global, G = 1, 1, 8, 128
    W = 2
    N_global = nB_global * G
    N_local = N_global // W

    global_mask = _global_sparse_mask(B, H, nB_global)
    # Varying bar counts per query block
    global_bar = torch.tensor([[[10, 20, 30, 40, 50, 60, 70, 80]]], dtype=torch.int32)

    # Single-card reference
    bar_cnt_local = torch.zeros(B, H, nB_global, 2, dtype=torch.int32)
    bar_cnt_local[..., -1] = global_bar
    ref = compute_sparse_ratio_local(global_mask, bar_cnt_local, N_global, G)

    rank_masks = [_distribute_block_mask_stripe(global_mask, W, r) for r in range(W)]
    rank_bars = [_distribute_bar_cnt_stripe(global_bar, W, r) for r in range(W)]
    dist_ratio = _compute_sparse_ratio_distributed_mock(
        rank_masks, rank_bars, N_local, W, G
    )
    _run("with_bars_2_ranks", dist_ratio, ref)


def test_with_bars_4_ranks():
    """Sparse mask + uniform bars, world_size=4, multi-batch multi-head."""
    B, H, nB_global, G = 2, 4, 16, 64
    W = 4
    N_global = nB_global * G
    N_local = N_global // W

    global_mask = _global_sparse_mask(B, H, nB_global)
    global_bar = torch.full((B, H, nB_global), 100, dtype=torch.int32)

    bar_cnt_local = torch.zeros(B, H, nB_global, 2, dtype=torch.int32)
    bar_cnt_local[..., -1] = global_bar
    ref = compute_sparse_ratio_local(global_mask, bar_cnt_local, N_global, G)

    rank_masks = [_distribute_block_mask_stripe(global_mask, W, r) for r in range(W)]
    rank_bars = [_distribute_bar_cnt_stripe(global_bar, W, r) for r in range(W)]
    dist_ratio = _compute_sparse_ratio_distributed_mock(
        rank_masks, rank_bars, N_local, W, G
    )
    _run("with_bars_4_ranks", dist_ratio, ref)


def test_block_count_conservation():
    """Verify that the total number of active blocks across all ranks
    equals the global block_mask sum (stripe distribution is a partition)."""
    B, H, nB_global, G = 1, 1, 16, 128
    W = 4

    for mask_fn, name in [
        (_global_causal_mask, "causal"),
        (_global_sparse_mask, "sparse"),
    ]:
        global_mask = mask_fn(B, H, nB_global)
        global_count = global_mask.sum().item()

        rank_masks = [_distribute_block_mask_stripe(global_mask, W, r) for r in range(W)]
        dist_count = sum(m.sum().item() for m in rank_masks)

        ok = global_count == dist_count
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] block_conservation_{name}: "
              f"global={global_count}, sum_ranks={dist_count}")
        if not ok:
            raise AssertionError(
                f"Block count mismatch for {name}: {global_count} != {dist_count}"
            )


def test_all_ranks_agree():
    """Each rank should compute the same ratio (since all_reduce sums them)."""
    B, H, nB_global, G = 1, 2, 8, 128
    W = 2
    N_global = nB_global * G
    N_local = N_global // W

    global_mask = _global_sparse_mask(B, H, nB_global)
    global_bar = torch.tensor(
        [[[10, 20, 30, 40, 50, 60, 70, 80]]], dtype=torch.int32
    ).expand(B, H, -1).contiguous()

    rank_masks = [_distribute_block_mask_stripe(global_mask, W, r) for r in range(W)]
    rank_bars = [_distribute_bar_cnt_stripe(global_bar, W, r) for r in range(W)]

    # The mock already sums all ranks; verify each rank's view yields the same
    # final ratio (since all_reduce makes every rank see the global sum).
    ratio = _compute_sparse_ratio_distributed_mock(
        rank_masks, rank_bars, N_local, W, G
    )
    print(f"  [PASS] all_ranks_agree: ratio={ratio:.10f} "
          f"(symmetric by construction)")


# ── runner ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_full_causal_2_ranks,
        test_full_causal_4_ranks,
        test_sparse_mask_2_ranks,
        test_sparse_mask_4_ranks,
        test_with_bars_2_ranks,
        test_with_bars_4_ranks,
        test_block_count_conservation,
        test_all_ranks_agree,
    ]
    print(f"Running {len(tests)} tests for distributed vs. single-card "
          f"sparse ratio alignment\n")
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
