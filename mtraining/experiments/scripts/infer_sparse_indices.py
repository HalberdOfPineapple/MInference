#!/usr/bin/env python3
# Copyright (c) 2026 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""Single-card inference script for collecting sparse attention data.

Loads a merged checkpoint, applies MInference sparse attention
(single-card ``minference_flash_attn_func``), runs forward passes over
a random subset of dataset samples (selected with a fixed seed), and
records per-layer sparse attention artefacts.

Each data category is controlled by a separate environment variable /
CLI flag so that expensive tensors can be skipped:

- **Indices** (``COLLECT_SPARSE_INDEX=1`` / ``--collect_indices``):
  Vertical and slash index tensors (v_idx, s_idx).
- **Block masks** (``COLLECT_BLOCK_MASK=1`` / ``--collect_block_mask``):
  Block mask and bar count tensors — very large for long contexts.
- **Sparse ratios** are always collected when any flag is active (tiny).

Output layout::

    <output_dir>/
        indices/sample_0000.pt   # {layer: {"v_idx": …, "s_idx": …}}
        masks/sample_0000.pt     # {layer: {"block_mask": …, "bar_cnt": …}}
        sparse_ratios.json       # {sample: {layer: ratio}}
        sample_meta.json

Usage::

    python infer_sparse_indices.py \\
        --model_id Qwen/Qwen2.5-3B \\
        --model_config_path ../../model_configs/qwen2/lc_config_3B \\
        --ckpt_path /path/to/merged_ckpts/0000-0001/pytorch_model.bin \\
        --pattern_config Qwen2.5_3B_flex_0.90 \\
        --dataset_path /path/to/processed_dataset \\
        --output_dir /path/to/output \\
        --num_samples 20 --seed 42 \\
        --collect_indices --collect_block_mask
"""

import argparse
import json
import os
import random
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

    print(f"Creating model {args.model_id} with config from {args.model_config_path or args.model_id}", flush=True)
    model = model_cls.from_pretrained(
        args.model_id,
        config=model_config,
        torch_dtype=torch.bfloat16,
    )
    if not args.ckpt_path:
        raise ValueError("Checkpoint path must be provided to load merged weights for index collection.")
    
    if '0000-0000' in args.ckpt_path:
        print(f"Using un-trained model for testing", flush=True)
        return model, model_config
    


    # Load merged checkpoint weights
    print(f"Loading checkpoint from {args.ckpt_path}", flush=True)
    state_dict = torch.load(args.ckpt_path, map_location="cpu")

    # Handle nested dict (full checkpoint vs raw weights)
    if "model" in state_dict and not any("." in k for k in list(state_dict.keys())[:5]):
        state_dict = state_dict["model"]

    # Detect key format and convert to the model's expected dotted keys.
    #
    # The merge_ckpts post-processing (merge_utils.py) saves a flat dict
    # with nnscaler underscore keys stripped of their "model_" prefix,
    # e.g. "layers_0_self_attn_q_proj_weight".  The model expects dotted
    # keys like "model.layers.0.self_attn.q_proj.weight".
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    if ckpt_keys & model_keys:
        # Keys already in the correct format — use as-is
        pass
    elif not any("." in k for k in list(ckpt_keys)[:5]):
        # Flat underscore keys from nnscaler merge post-processing.
        # Build a reverse map: underscore_key -> dotted_key, then match
        # checkpoint keys (possibly with the stripped "model_" prefix).
        key_map = {mk.replace(".", "_"): mk for mk in model_keys}
        new_state_dict = {}
        unmapped = []
        for fk, v in state_dict.items():
            if fk in key_map:
                new_state_dict[key_map[fk]] = v
            elif f"model_{fk}" in key_map:
                new_state_dict[key_map[f"model_{fk}"]] = v
            else:
                unmapped.append(fk)
        if unmapped:
            print(f"[WARNING] Could not map {len(unmapped)} checkpoint keys: {unmapped[:5]}", flush=True)
        state_dict = new_state_dict
    else:
        # Dotted keys — only strip "model." prefix if the keys don't
        # already match (e.g. double "model.model." prefix).
        first_key = next(iter(state_dict))
        if first_key.startswith("model.") and first_key not in model_keys:
            state_dict = {k[len("model."):]: v for k, v in state_dict.items()}

    result = model.load_state_dict(state_dict, strict=False)
    num_loaded = len(model_keys) - len(result.missing_keys)
    if num_loaded == 0:
        raise RuntimeError(
            f"No checkpoint weights were loaded! "
            f"First 3 ckpt keys: {list(state_dict.keys())[:3]}, "
            f"First 3 model keys: {list(model_keys)[:3]}"
        )
    print(f"Checkpoint loaded: {num_loaded}/{len(model_keys)} parameters updated", flush=True)
    if result.missing_keys:
        print(f"[WARNING] Missing keys ({len(result.missing_keys)}): {result.missing_keys[:5]}", flush=True)
    if result.unexpected_keys:
        print(f"[WARNING] Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]}", flush=True)

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
    print(f"Pattern config: {pattern_config_path}", flush=True)

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
    print(f"Applied MInference patching (implementation=default, granularity={args.granularity})", flush=True)


def run_inference(model, dataset, args):
    """Run forward passes and collect sparse attention data."""
    collector = get_index_collector()
    if not collector.enabled:
        raise RuntimeError(
            "IndexCollector is not enabled. Set COLLECT_SPARSE_INDEX=1 and/or "
            "COLLECT_BLOCK_MASK=1, or use --collect_indices / --collect_block_mask."
        )

    # Select a random subset of samples with a fixed seed for reproducibility
    total_dataset_size = len(dataset)
    num_samples = min(args.num_samples, total_dataset_size)
    rng = random.Random(args.seed)
    selected_indices = sorted(rng.sample(range(total_dataset_size), num_samples))
    print(
        f"Selected {num_samples}/{total_dataset_size} samples with seed={args.seed}: "
        f"{selected_indices[:10]}{'...' if num_samples > 10 else ''}",
        flush=True,
    )

    # Save the selected indices for reproducibility
    os.makedirs(args.output_dir, exist_ok=True)
    meta = {"seed": args.seed, "num_samples": num_samples, "selected_indices": selected_indices}
    with open(os.path.join(args.output_dir, "sample_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    model.eval()
    model.cuda()

    for i, dataset_idx in enumerate(selected_indices):
        input_ids = torch.tensor(
            dataset[dataset_idx]["input_ids"], dtype=torch.long
        ).unsqueeze(0).cuda()
        seq_len = input_ids.shape[1]

        print(f"[{i+1}/{num_samples}] dataset_idx={dataset_idx}, seq_len={seq_len}", flush=True)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model.model(input_ids=input_ids, use_cache=False, return_dict=False)

        collector.finish_sample()

        # Save incrementally to avoid losing data on OOM
        if (i + 1) % args.save_interval == 0:
            collector.save(args.output_dir)
            print(f"Saved data up to sample {i}", flush=True)

    # Final save
    collector.save(args.output_dir)
    print(f"Done. Data saved to {args.output_dir}", flush=True)


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
        "--seed", type=int, default=42,
        help="Random seed for selecting dataset samples (default: 42)",
    )
    parser.add_argument(
        "--granularity", type=int, default=128,
        help="Block size for sparse attention (default: 128)",
    )
    parser.add_argument(
        "--save_interval", type=int, default=5,
        help="Save indices to disk every N samples",
    )
    parser.add_argument(
        "--collect_indices", action="store_true", default=False,
        help="Collect vertical/slash index tensors (v_idx, s_idx). "
             "Equivalent to env COLLECT_SPARSE_INDEX=1.",
    )
    parser.add_argument(
        "--collect_block_mask", action="store_true", default=False,
        help="Collect block_mask and bar_cnt tensors (very large for long contexts). "
             "Equivalent to env COLLECT_BLOCK_MASK=1.",
    )
    args = parser.parse_args()

    # Set collection env vars from CLI flags.
    # If neither flag is given, default to collecting indices only
    # (backward-compatible lightweight default).
    if not args.collect_indices and not args.collect_block_mask:
        args.collect_indices = True
    if args.collect_indices:
        os.environ["COLLECT_SPARSE_INDEX"] = "1"
    if args.collect_block_mask:
        os.environ["COLLECT_BLOCK_MASK"] = "1"

    model, model_config = load_model(args)
    apply_minference_patching(model, args)

    print(f"Loading dataset from {args.dataset_path}", flush=True)
    dataset = load_from_disk(args.dataset_path)

    run_inference(model, dataset, args)


if __name__ == "__main__":
    main()
