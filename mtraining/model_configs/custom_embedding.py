# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from typing import Optional

import torch
from nnscaler.graph.parser.register import register_op
from torch import Tensor, nn
from torch.overrides import handle_torch_function, has_torch_function_variadic

DEVICE_TYPE = "CUDA" if not torch.version.hip else "HIP"
MAX_CHUNK = 32 * 1024


def chunked_embedding(
    input_tensor: Tensor,
    weight: Tensor,
    padding_idx: Optional[int] = None,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
):
    pieces = []
    for seq_slice in torch.split(input_tensor, MAX_CHUNK, dim=1):
        out_part = torch.embedding(
            weight,
            seq_slice,  # [B, MAX_CHUNK]
            padding_idx or -1,
            scale_grad_by_freq,
            sparse,
        )
        pieces.append(out_part)

    cat_embedding = torch.cat(pieces, dim=1)
    return cat_embedding


def custom_embedding(
    input: Tensor,  # [B, N]
    weight: Tensor,  # [V, D]
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    if has_torch_function_variadic(input, weight):
        return handle_torch_function(
            custom_embedding,
            (input, weight),
            input,
            weight,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )

    if padding_idx is not None:
        if padding_idx > 0:
            assert padding_idx < weight.size(
                0
            ), "Padding_idx must be within num_embeddings"
        elif padding_idx < 0:
            assert padding_idx >= -weight.size(
                0
            ), "Padding_idx must be within num_embeddings"
            padding_idx = weight.size(0) + padding_idx
    else:
        padding_idx = -1

    if max_norm is not None:
        input = input.contiguous()
        torch.embedding_renorm_(weight.detach(), input, max_norm, norm_type)

    # if DEVICE_TYPE == "HIP":
    #     res = chunked_embedding(input, weight, padding_idx, scale_grad_by_freq, sparse)
    #     return res
    # else:
    #     return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
    res = chunked_embedding(input, weight, padding_idx, scale_grad_by_freq, sparse)
    return res


class CustomEmbedding(nn.Embedding):
    def forward(self, input: Tensor) -> Tensor:
        print(
            f"{self.__class__.__name__} | input_ids dtype: {input.dtype}, shape: {input.shape}"
        )
        return custom_embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


register_op("b l, v^ d^ -> b l d^")(custom_embedding)
