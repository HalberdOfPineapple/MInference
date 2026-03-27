#!/usr/bin/env python3
# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""Single-card inference script for collecting sparse attention indices.

Loads a merged checkpoint, applies MInference sparse attention
(single-card ``minference_flash_attn_func``), runs forward passes over
dataset samples, and records the vertical / slash index tensors selected
by each attention head in every layer.

Usage::

    COLLECT_SPARSE_INDEX=1 python infer_sparse_indices.py \
        --model_id Qwen/Qwen2.5-3B \
        --model_config_path ../../model_configs/qwen2/lc_config_3B \
        --ckpt_path /path/to/merged_ckpts/0000-0001/pytorch_model.bin \
        --pattern_config Qwen2.5_3B_flex_0.90 \
        --dataset_path /path/to/processed_dataset \
        --output_dir /path/to/output \
        --num_samples 20
"""

import argparse
import json
import logging
import os
import sys

import torch
from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so mtraining / minference are importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from minference.configs.model2path import BASE_DIR as SPARSE_PATTERN_CONFIG_DIR
from minference.dist_ops.index_collector import get_index_collector
from mtraining.attn_funcs import AttnType, overwrite_attn_implementation
from mtraining.attn_funcs.minfer_func import MInferAttnFunc
from mtraining.model_configs import get_model_attn_funcs, get_model_cls

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(args):
    """Load model with long-context config and merged checkpoint weights."""
    model_cls = get_model_cls(args.model_id)

    # Load config — use custom long-context config if provided
    if args.model_config_path:
        model_config = AutoConfig.from_pretrained(
            args.model_config_path, trust_remote_code=True
        )
    else:
        model_config = AutoConfig.from_pretrained(
            args.model_id, trust_remote_code=True
        )
    model_config._attn_implementation = "flash_attention_2"

    logger.info(f"Creating model {args.model_id} with config from {args.model_config_path or args.model_id}")
    model = model_cls.from_pretrained(
        args.model_id,
        config=model_config,
        torch_dtype=torch.bfloat16,
    )
    if not args.ckpt_path:
        raise ValueError("Checkpoint path must be provided to load merged weights for index collection.")
    
    if '0000-0000' in args.ckpt_path:
        logger.info(f"Using un-trained model for testing")
        return model, model_config
    


    # Load merged checkpoint weights
    logger.info(f"Loading checkpoint from {args.ckpt_path}")
    state_dict = torch.load(args.ckpt_path, map_location="cpu")

    # Handle nested dict (full checkpoint vs raw weights)
    if "model" in state_dict and not any("." in k for k in list(state_dict.keys())[:5]):
        state_dict = state_dict["model"]

    # Strip "model." prefix if present (from merge_checkpoint)
    first_key = next(iter(state_dict))
    if first_key.startswith("model."):
        state_dict = {k[len("model."):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    logger.info("Checkpoint loaded successfully")
    return model, model_config


def apply_minference_patching(model, args):
    """Attach MInferAttnFunc to every attention layer for single-card sparse attention."""
    # Overwrite attention implementation to use minfer_attention_forward
    attn_dict = get_model_attn_funcs(args.model_id)
    overwrite_attn_implementation(attn_dict, AttnType.MINFER)

    # Resolve pattern config path
    pattern_config_path = os.path.join(
        SPARSE_PATTERN_CONFIG_DIR,
        f"{args.pattern_config}.json",
    )
    if not os.path.exists(pattern_config_path):
        raise FileNotFoundError(f"Pattern config not found: {pattern_config_path}")
    logger.info(f"Pattern config: {pattern_config_path}")

    # Attach MInferAttnFunc to each attention layer (implementation="default" for single-card)
    Attention = model.model.layers[0].self_attn.__class__

    def update_module(m):
        if isinstance(m, Attention):
            m.minfer_attn_func = MInferAttnFunc()
            m.minfer_attn_func.init_minfer_params(
                config_path=pattern_config_path,
                minfer_implementation="default",
                granularity=args.granularity,
            )

    model.apply(update_module)
    logger.info(f"Applied MInference patching (implementation=default, granularity={args.granularity})")


def run_inference(model, dataset, args):
    """Run forward passes and collect sparse attention indices."""
    collector = get_index_collector()
    if not collector.enabled:
        raise RuntimeError(
            "IndexCollector is not enabled. Set COLLECT_SPARSE_INDEX=1 environment variable."
        )

    num_samples = min(args.num_samples, len(dataset))
    logger.info(f"Running inference on {num_samples} samples (seq_len from dataset)")

    model.eval()
    model.cuda()

    for sample_idx in range(num_samples):
        input_ids = torch.tensor(
            dataset[sample_idx]["input_ids"], dtype=torch.long
        ).unsqueeze(0).cuda()
        seq_len = input_ids.shape[1]

        logger.info(f"Sample {sample_idx}/{num_samples}: seq_len={seq_len}")

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model.model(input_ids=input_ids, use_cache=False, return_dict=False)

        collector.finish_sample()

        # Save incrementally to avoid losing data on OOM
        if (sample_idx + 1) % args.save_interval == 0:
            collector.save(args.output_dir)
            logger.info(f"Saved indices up to sample {sample_idx}")

    # Final save
    collector.save(args.output_dir)
    logger.info(f"Done. Indices saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Single-card sparse attention index collection via inference"
    )
    parser.add_argument(
        "--model_id", type=str, default="Qwen/Qwen2.5-3B",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--model_config_path", type=str, default=None,
        help="Path to custom model config directory (e.g., model_configs/qwen2/lc_config_3B)",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None,
        help="Path to merged checkpoint file (pytorch_model.bin)",
    )
    parser.add_argument(
        "--pattern_config", type=str, default="Qwen2.5_3B_flex_0.90",
        help="Name of the MInference pattern config (without .json)",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to pre-processed HuggingFace dataset on disk",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save collected index tensors",
    )
    parser.add_argument(
        "--num_samples", type=int, default=20,
        help="Number of dataset samples to process",
    )
    parser.add_argument(
        "--granularity", type=int, default=128,
        help="Block size for sparse attention (default: 128)",
    )
    parser.add_argument(
        "--save_interval", type=int, default=5,
        help="Save indices to disk every N samples",
    )
    args = parser.parse_args()

    # Ensure collection is enabled
    os.environ["COLLECT_SPARSE_INDEX"] = "1"

    model, model_config = load_model(args)
    apply_minference_patching(model, args)

    logger.info(f"Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)

    run_inference(model, dataset, args)


if __name__ == "__main__":
    main()
