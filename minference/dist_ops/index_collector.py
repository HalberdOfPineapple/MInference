# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import os
from typing import Dict, Optional, Tuple

import torch


class IndexCollector:
    """Collect per-(sample, layer) sparse attention data during inference.

    Each data type is controlled by a separate environment variable so that
    expensive tensors (block masks) can be skipped when not needed:

    - ``COLLECT_SPARSE_INDEX=1`` — record vertical / slash index tensors.
    - ``COLLECT_BLOCK_MASK=1``   — record block_mask and bar_cnt tensors
      (these are very large for long-context models).

    Sparse ratios (a single float per layer) are always recorded when the
    collector is enabled (i.e. any of the above flags is set).

    Collected data is saved into separate subdirectories so that each data
    type can be managed independently:

    - ``<output_dir>/indices/sample_XXXX.pt``
    - ``<output_dir>/masks/sample_XXXX.pt``
    - ``<output_dir>/sparse_ratios.json``
    """

    def __init__(self):
        self.collect_indices: bool = os.getenv("COLLECT_SPARSE_INDEX", "0") == "1"
        self.collect_masks: bool = os.getenv("COLLECT_BLOCK_MASK", "0") == "1"
        self.enabled: bool = self.collect_indices or self.collect_masks

        # Per-layer storage for the *current* sample being processed.
        self._current_indices: Dict[int, Dict[str, torch.Tensor]] = {}
        self._current_masks: Dict[int, Dict[str, torch.Tensor]] = {}
        self._layer_counter: int = 0

        # Accumulated across samples: sample_idx -> {layer_idx -> {...}}
        self._all_indices: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
        self._all_masks: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
        self._sample_counter: int = 0

        # Sparse ratio records: {(sample_idx, layer_idx): float}
        self._sparse_ratios: Dict[Tuple[int, int], float] = {}
        self._ratio_layer_counter: int = 0

        if self.enabled:
            flags = []
            if self.collect_indices:
                flags.append("indices")
            if self.collect_masks:
                flags.append("masks")
            print(
                f"{__name__} | IndexCollector enabled "
                f"(collecting: {', '.join(flags)}, sparse_ratios)"
            )

    def record(self, v_idx: torch.Tensor, s_idx: torch.Tensor) -> None:
        """Store vertical and slash indices for the current layer.

        The layer counter is always incremented (when the collector is
        enabled) so that ``record_mask`` can reference the same layer,
        but data is only stored when ``collect_indices`` is True.

        Args:
            v_idx: Vertical indices, shape ``[batch_size, num_heads, max_v_size]``.
            s_idx: Slash indices, shape ``[batch_size, num_heads, max_s_size]``.
        """
        if not self.enabled:
            return
        layer = self._layer_counter
        if self.collect_indices:
            self._current_indices[layer] = {
                "v_idx": v_idx.squeeze(0).cpu(),
                "s_idx": s_idx.squeeze(0).cpu(),
            }
        self._layer_counter += 1

    def record_mask(
        self, block_mask: torch.Tensor, bar_cnt: torch.Tensor
    ) -> None:
        """Store block mask and bar count for the current layer.

        Called from ``build_index_local`` after ``convert_indices``.
        Data is only stored when ``collect_masks`` is True.

        Args:
            block_mask: ``[batch_size, num_heads, num_blocks, num_blocks]`` (bool).
            bar_cnt: ``[batch_size, num_heads, num_blocks, 2]`` (int32).
        """
        if not self.enabled or not self.collect_masks:
            return
        # The layer counter was already incremented by record(), so use
        # the previous layer index.
        layer = self._layer_counter - 1
        self._current_masks[layer] = {
            "block_mask": block_mask.squeeze(0).cpu(),
            "bar_cnt": bar_cnt.squeeze(0).cpu(),
        }

    def record_sparse_ratio(self, sparse_ratio: float) -> None:
        """Store the sparse ratio for the current layer.

        Called from ``build_index_local`` after the block/bar masks are built.
        Always recorded when the collector is enabled (ratios are tiny).
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
        if not self.enabled:
            return
        has_data = (
            bool(self._current_indices)
            or bool(self._current_masks)
            or bool(self._sparse_ratios)
        )
        if not has_data:
            return
        if self._current_indices:
            self._all_indices[self._sample_counter] = self._current_indices
        if self._current_masks:
            self._all_masks[self._sample_counter] = self._current_masks
        self._current_indices = {}
        self._current_masks = {}
        self._layer_counter = 0
        self._ratio_layer_counter = 0
        self._sample_counter += 1

    def save(self, output_dir: str) -> None:
        """Save collected data to disk.

        Each data type is written to its own subdirectory / file:

        - ``indices/sample_XXXX.pt`` — ``{layer_idx: {"v_idx", "s_idx"}}``
        - ``masks/sample_XXXX.pt``   — ``{layer_idx: {"block_mask", "bar_cnt"}}``
        - ``sparse_ratios.json``     — ``{sample_idx: {layer_idx: ratio}}``
        """
        os.makedirs(output_dir, exist_ok=True)

        if self._all_indices:
            indices_dir = os.path.join(output_dir, "indices")
            os.makedirs(indices_dir, exist_ok=True)
            for sample_idx, layers in self._all_indices.items():
                path = os.path.join(indices_dir, f"sample_{sample_idx:04d}.pt")
                torch.save(layers, path)
            print(
                f"{__name__} | Saved {len(self._all_indices)} index sample(s) "
                f"to {indices_dir}"
            )

        if self._all_masks:
            masks_dir = os.path.join(output_dir, "masks")
            os.makedirs(masks_dir, exist_ok=True)
            for sample_idx, layers in self._all_masks.items():
                path = os.path.join(masks_dir, f"sample_{sample_idx:04d}.pt")
                torch.save(layers, path)
            print(
                f"{__name__} | Saved {len(self._all_masks)} mask sample(s) "
                f"to {masks_dir}"
            )

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
        self._current_indices = {}
        self._current_masks = {}
        self._layer_counter = 0
        self._ratio_layer_counter = 0
        self._all_indices = {}
        self._all_masks = {}
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
