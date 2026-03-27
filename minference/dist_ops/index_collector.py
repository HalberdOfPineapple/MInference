# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import os
from typing import Dict, Optional, Tuple

import torch


class IndexCollector:
    """Collect per-(sample, layer) sparse attention data during inference.

    Enabled by setting the environment variable ``COLLECT_SPARSE_INDEX=1``.
    Records for every layer in a forward pass:
    - Vertical and slash index tensors (from ``calc_index_local``)
    - Block mask and bar count tensors (from ``convert_indices``)
    - Sparse ratio (derived from the mask tensors)
    """

    def __init__(self):
        self.enabled: bool = os.getenv("COLLECT_SPARSE_INDEX", "0") == "1"
        # layer_idx -> {"v_idx": Tensor, "s_idx": Tensor,
        #               "block_mask": Tensor, "bar_cnt": Tensor}
        self._current_sample: Dict[int, Dict[str, torch.Tensor]] = {}
        self._layer_counter: int = 0
        # sample_idx -> {layer_idx -> {...}}
        self._all_samples: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
        self._sample_counter: int = 0

        # Sparse ratio records: {(sample_idx, layer_idx): float}
        self._sparse_ratios: Dict[Tuple[int, int], float] = {}
        self._ratio_layer_counter: int = 0

        if self.enabled:
            print(f"{__name__} | IndexCollector enabled")

    def record(self, v_idx: torch.Tensor, s_idx: torch.Tensor) -> None:
        """Store vertical and slash indices for the current layer.

        Args:
            v_idx: Vertical indices, shape ``[batch_size, num_heads, max_v_size]``.
            s_idx: Slash indices, shape ``[batch_size, num_heads, max_s_size]``.
        """
        if not self.enabled:
            return
        layer = self._layer_counter
        if layer not in self._current_sample:
            self._current_sample[layer] = {}
        # Remove batch dim (batch_size is expected to be 1 during inference)
        self._current_sample[layer]["v_idx"] = v_idx.squeeze(0).cpu()
        self._current_sample[layer]["s_idx"] = s_idx.squeeze(0).cpu()
        self._layer_counter += 1

    def record_mask(
        self, block_mask: torch.Tensor, bar_cnt: torch.Tensor
    ) -> None:
        """Store block mask and bar count for the current layer.

        Called from ``build_index_local`` after ``convert_indices``.

        Args:
            block_mask: ``[batch_size, num_heads, num_blocks, num_blocks]`` (bool).
            bar_cnt: ``[batch_size, num_heads, num_blocks, 2]`` (int32).
        """
        if not self.enabled:
            return
        # The layer counter was already incremented by record(), so use
        # the previous layer index.
        layer = self._layer_counter - 1
        if layer not in self._current_sample:
            self._current_sample[layer] = {}
        self._current_sample[layer]["block_mask"] = block_mask.squeeze(0).cpu()
        self._current_sample[layer]["bar_cnt"] = bar_cnt.squeeze(0).cpu()

    def record_sparse_ratio(self, sparse_ratio: float) -> None:
        """Store the sparse ratio for the current layer.

        Called from ``build_index_local`` after the block/bar masks are built.
        """
        if not self.enabled:
            return
        key = (self._sample_counter, self._ratio_layer_counter)
        self._sparse_ratios[key] = sparse_ratio
        self._ratio_layer_counter += 1

    def finish_sample(self) -> None:
        """Mark the end of a single sample's forward pass.

        Stores the accumulated layer data and resets for the next sample.
        """
        if not self.enabled or not self._current_sample:
            return
        self._all_samples[self._sample_counter] = self._current_sample
        self._current_sample = {}
        self._layer_counter = 0
        self._ratio_layer_counter = 0
        self._sample_counter += 1

    def save(self, output_dir: str) -> None:
        """Save collected data to disk.

        Per-sample ``.pt`` files, each mapping ``layer_idx`` to a dict with
        keys ``v_idx``, ``s_idx``, ``block_mask``, ``bar_cnt``.

        Sparse ratios: ``sparse_ratios.json`` with structure
        ``{sample_idx: {layer_idx: ratio}}``.
        """
        if not self._all_samples:
            return
        os.makedirs(output_dir, exist_ok=True)
        for sample_idx, layers in self._all_samples.items():
            path = os.path.join(output_dir, f"sample_{sample_idx:04d}.pt")
            torch.save(layers, path)
        print(
            f"{__name__} | Saved {len(self._all_samples)} sample(s) "
            f"to {output_dir}"
        )

        # Save sparse ratios
        if self._sparse_ratios:
            import json
            payload: Dict[str, Dict[str, float]] = {}
            for (si, li), ratio in sorted(self._sparse_ratios.items()):
                payload.setdefault(str(si), {})[str(li)] = ratio
            path = os.path.join(output_dir, "sparse_ratios.json")
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"{__name__} | Saved sparse ratios to {path}")

    def reset(self) -> None:
        """Clear all collected data."""
        self._current_sample = {}
        self._layer_counter = 0
        self._ratio_layer_counter = 0
        self._all_samples = {}
        self._sample_counter = 0
        self._sparse_ratios = {}


_INDEX_COLLECTOR: Optional[IndexCollector] = None


def get_index_collector() -> IndexCollector:
    """Return the global singleton ``IndexCollector``, creating on first call."""
    global _INDEX_COLLECTOR
    if _INDEX_COLLECTOR is None:
        _INDEX_COLLECTOR = IndexCollector()
    return _INDEX_COLLECTOR


def compute_sparse_ratio_local(
    block_mask: torch.Tensor,
    bar_cnt: torch.Tensor,
    num_tokens: int,
    granularity: int,
) -> float:
    """Compute sparse ratio from single-card ``build_index_local`` outputs.

    This is the single-machine counterpart of
    :func:`minference.dist_ops.utils.compute_sparse_ratio`, adapted for the
    output shapes of ``build_index_local`` (world_size=1, rank=0):

    - ``block_mask``: ``[batch_size, num_heads, num_blocks, num_blocks]``
      (the world_size dim has already been indexed away by
      ``block_mask = block_mask[rank]``).
    - ``bar_cnt``: ``[batch_size, num_heads, num_blocks, 2]``
      (world_size+1 = 2 when world_size=1).  The last column (index -1)
      holds the cumulative count of selected bar entries per query block.

    The distributed version differs in three ways:

    1. ``block_mask`` has an extra leading ``[world_size]`` dimension because
       each ring-attention step produces a separate mask for the K/V shard
       it currently holds.
    2. ``bar_cnt`` has ``world_size+1`` columns (cumulative per rank); only
       the last column gives the total count.
    3. Both sums are reduced across ranks with ``dist.all_reduce``.

    Here none of those apply — world_size=1 so there is one mask and no
    communication.

    Returns:
        Sparse ratio in [0, 1].  Higher = sparser (fewer entries computed).
    """
    batch_size, num_heads = block_mask.shape[:2]
    num_blocks = block_mask.shape[2]
    block_area = granularity * granularity

    # Active block entries (dense blocks in the block-sparse attention)
    num_active_block_entries = block_mask.sum().item() * block_area
    # Causal correction: each block on the diagonal is roughly half-used
    num_active_block_entries -= num_blocks * block_area * batch_size * num_heads / 2.0

    # Active bar entries (fine-grained vertical+slash indices outside blocks)
    num_active_bar_entries = bar_cnt[..., -1].sum().item() * granularity

    num_active_entries = num_active_block_entries + num_active_bar_entries
    total_entries = num_tokens * num_tokens * batch_size * num_heads / 2.0
    sparse_ratio = 1.0 - num_active_entries / total_entries
    return sparse_ratio
