# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import os
from typing import Dict, Optional, Tuple

import torch


class IndexCollector:
    """Collect per-(sample, layer) sparse attention indices during inference.

    Enabled by setting the environment variable ``COLLECT_SPARSE_INDEX=1``.
    Records the vertical (``v_idx``) and slash (``s_idx``) index tensors
    produced by ``calc_index_local`` for every layer in a forward pass.
    """

    def __init__(self):
        self.enabled: bool = os.getenv("COLLECT_SPARSE_INDEX", "0") == "1"
        # layer_idx -> {"v_idx": Tensor, "s_idx": Tensor}
        self._current_sample: Dict[int, Dict[str, torch.Tensor]] = {}
        self._layer_counter: int = 0
        # sample_idx -> {layer_idx -> {"v_idx": Tensor, "s_idx": Tensor}}
        self._all_samples: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
        self._sample_counter: int = 0
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
        # Remove batch dim (batch_size is expected to be 1 during inference)
        self._current_sample[self._layer_counter] = {
            "v_idx": v_idx.squeeze(0).cpu(),
            "s_idx": s_idx.squeeze(0).cpu(),
        }
        self._layer_counter += 1

    def finish_sample(self) -> None:
        """Mark the end of a single sample's forward pass.

        Stores the accumulated layer indices and resets for the next sample.
        """
        if not self.enabled or not self._current_sample:
            return
        self._all_samples[self._sample_counter] = self._current_sample
        self._current_sample = {}
        self._layer_counter = 0
        self._sample_counter += 1

    def save(self, output_dir: str) -> None:
        """Save collected indices to disk as ``.pt`` files, one per sample.

        Each file contains a dict mapping ``layer_idx`` to
        ``{"v_idx": Tensor[num_heads, max_v_size],
          "s_idx": Tensor[num_heads, max_s_size]}``.
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

    def reset(self) -> None:
        """Clear all collected data."""
        self._current_sample = {}
        self._layer_counter = 0
        self._all_samples = {}
        self._sample_counter = 0


_INDEX_COLLECTOR: Optional[IndexCollector] = None


def get_index_collector() -> IndexCollector:
    """Return the global singleton ``IndexCollector``, creating on first call."""
    global _INDEX_COLLECTOR
    if _INDEX_COLLECTOR is None:
        _INDEX_COLLECTOR = IndexCollector()
    return _INDEX_COLLECTOR
