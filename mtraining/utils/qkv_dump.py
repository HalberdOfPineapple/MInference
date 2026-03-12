# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def _dist_is_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


class _QKVDumpManager:
    def __init__(self):
        self.enabled = _env_flag("MTRAIN_DUMP_QKV_ENABLED")
        self.output_root = Path(os.getenv("MTRAIN_DUMP_QKV_ROOT", "./qkv_dump"))
        self.max_samples = int(os.getenv("MTRAIN_DUMP_QKV_MAX_SAMPLES", "50"))
        self.writer_global_rank = int(os.getenv("MTRAIN_DUMP_QKV_RANK", "0"))
        self.overwrite = _env_flag("MTRAIN_DUMP_QKV_OVERWRITE")
        self.rank = dist.get_rank() if _dist_is_ready() else int(os.getenv("RANK", "0"))
        self.world_size = dist.get_world_size() if _dist_is_ready() else 1

        self.current_batch_sample_indices: List[int] = []
        self.next_sample_idx = 0
        self.max_layer_idx_seen = -1
        self.last_layer_idx: Optional[int] = None
        self._has_logged_status = False
        self._metadata_written = False

    @property
    def is_active(self) -> bool:
        return self.enabled and self.max_samples > 0

    def _maybe_start_batch(self, layer_idx: int, batch_size: int):
        if not self.is_active:
            self.current_batch_sample_indices = []
            return

        should_start = False
        if not self.current_batch_sample_indices:
            should_start = layer_idx == 0
        elif layer_idx == 0 and self.last_layer_idx == self.max_layer_idx_seen:
            should_start = True

        self.max_layer_idx_seen = max(self.max_layer_idx_seen, layer_idx)
        self.last_layer_idx = layer_idx

        if not should_start:
            return

        remaining = self.max_samples - self.next_sample_idx
        num_to_dump = max(0, min(batch_size, remaining))
        self.current_batch_sample_indices = list(
            range(self.next_sample_idx, self.next_sample_idx + num_to_dump)
        )
        self.next_sample_idx += num_to_dump

        if not self.current_batch_sample_indices:
            return

        self.output_root.mkdir(parents=True, exist_ok=True)
        if not self._has_logged_status:
            logger.info(
                "QKV dumping enabled on global rank %s/%s. Writer rank=%s, max samples=%s, output=%s",
                self.rank,
                self.world_size,
                self.writer_global_rank,
                self.max_samples,
                self.output_root,
            )
            self._has_logged_status = True

        if self.rank == self.writer_global_rank and not self._metadata_written:
            metadata = {
                "rank": self.rank,
                "writer_global_rank": self.writer_global_rank,
                "world_size": self.world_size,
                "max_samples": self.max_samples,
                "overwrite": self.overwrite,
            }
            with open(self.output_root / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            self._metadata_written = True

    def _gather_cpu_tensor(
        self, tensor: torch.Tensor, process_group: Optional[dist.ProcessGroup]
    ) -> torch.Tensor:
        cpu_tensor = tensor.detach().contiguous().to("cpu")
        if not _dist_is_ready():
            return cpu_tensor

        group_world_size = dist.get_world_size(group=process_group)
        if group_world_size == 1:
            return cpu_tensor

        gathered_tensors = [None for _ in range(group_world_size)]
        dist.all_gather_object(gathered_tensors, cpu_tensor, group=process_group)
        return torch.cat(gathered_tensors, dim=1)

    def dump_qkv(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        self._maybe_start_batch(layer_idx=layer_idx, batch_size=query_states.shape[0])
        if not self.current_batch_sample_indices:
            return

        query_states = self._gather_cpu_tensor(
            query_states, process_group=process_group
        )
        key_states = self._gather_cpu_tensor(key_states, process_group=process_group)
        value_states = self._gather_cpu_tensor(
            value_states, process_group=process_group
        )

        if self.rank != self.writer_global_rank:
            return

        for batch_idx, sample_idx in enumerate(self.current_batch_sample_indices):
            output_path = (
                self.output_root
                / f"layer_{layer_idx}"
                / f"sample_{sample_idx}"
                / "qkv.pt"
            )
            if output_path.exists() and not self.overwrite:
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "q": query_states[batch_idx],
                    "k": key_states[batch_idx],
                    "v": value_states[batch_idx],
                    "layer_idx": layer_idx,
                    "sample_idx": sample_idx,
                    "rank": self.rank,
                },
                output_path,
            )


_QKV_DUMP_MANAGER = _QKVDumpManager()


def maybe_dump_qkv(
    layer_idx: int,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    process_group: Optional[dist.ProcessGroup] = None,
):
    _QKV_DUMP_MANAGER.dump_qkv(
        layer_idx, query_states, key_states, value_states, process_group=process_group
    )
