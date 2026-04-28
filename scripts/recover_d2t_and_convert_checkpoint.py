#!/usr/bin/env python3
"""
Convert a speculators-dev Eagle3 training checkpoint to vLLM-compatible format.

The speculators-dev training framework uses a reduced draft vocabulary (default 32000
tokens) with a d2t mapping. The stored d2t.npy is typically in offset format
(target_idx = draft_idx + d2t[draft_idx]). This script recovers the absolute-index
mapping by matching checkpoint lm_head rows against the pre-trained Eagle3 vLLM
model's lm_head via hash-based exact lookup, then expands the lm_head to the full
target vocabulary size.

IMPORTANT: --vllm-model must be the PRE-TRAINED Eagle3 vLLM model (e.g.
lightseekorg-eagle3), NOT the verifier (K2.6). The spec rows are derived from the
Eagle3 model's lm_head; matching against the verifier (K2.6) will produce wrong results.

Usage:
    python3 recover_d2t_and_convert_checkpoint.py \\
        --checkpoint /data/training/my_run/17 \\
        --speculators-weights /data/models/my-eagle3-speculators.safetensors \\
        --vllm-model /data/models/lightseekorg-eagle3 \\
        --output /data/models/my-eagle3-ft
"""
import argparse
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def find_lm_head_shard(model_dir: str) -> str:
    """Find the safetensors shard containing lm_head.weight."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        import json
        with open(index_path) as f:
            index = json.load(f)
        shard = index["weight_map"]["lm_head.weight"]
        return os.path.join(model_dir, shard)
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        return single
    raise FileNotFoundError(f"Cannot find lm_head.weight in {model_dir}")


def recover_d2t(lm_spec: torch.Tensor, lm_eagle3: torch.Tensor) -> torch.Tensor:
    """
    Recover d2t absolute-index mapping by matching each spec row to an eagle3 row
    via hash lookup. Uses the first 16 float16 values as a hash key.

    lm_spec: [draft_vocab, hidden] — from pretrain speculators safetensors
    lm_eagle3: [full_vocab, hidden] — from pre-trained Eagle3 vLLM model (NOT verifier)

    Returns d2t: [draft_vocab] int64 with absolute target indices.
    """
    print("Building Eagle3 lm_head hash table...")
    eagle3_hash = {}
    for i in range(lm_eagle3.shape[0]):
        key = lm_eagle3[i, :16].half().cpu().numpy().tobytes()
        if key not in eagle3_hash:
            eagle3_hash[key] = i

    draft_vocab = lm_spec.shape[0]
    d2t = torch.zeros(draft_vocab, dtype=torch.long)
    n_found = 0
    for i in range(draft_vocab):
        key = lm_spec[i, :16].half().cpu().numpy().tobytes()
        if key in eagle3_hash:
            d2t[i] = eagle3_hash[key]
            n_found += 1
        else:
            d2t[i] = i  # fallback: identity

    n_missing = draft_vocab - n_found
    print(f"d2t recovered: {n_found}/{draft_vocab} exact matches, {n_missing} identity fallback")
    print(f"d2t unique targets: {d2t.unique().shape[0]}")
    return d2t


def convert(
    checkpoint_dir: str,
    speculators_weights: str,
    vllm_model_dir: str,
    output_dir: str,
) -> None:
    # Load reference lm_heads
    with safe_open(speculators_weights, framework="pt") as f:
        lm_spec = f.get_tensor("lm_head.weight")

    # Load pre-trained Eagle3 vLLM lm_head (NOT the verifier like K2.6)
    eagle3_shard = find_lm_head_shard(vllm_model_dir)
    with safe_open(eagle3_shard, framework="pt") as f:
        lm_eagle3 = f.get_tensor("lm_head.weight")

    print(f"speculators lm_head: {lm_spec.shape} {lm_spec.dtype}")
    print(f"Eagle3 vLLM lm_head: {lm_eagle3.shape} {lm_eagle3.dtype}")

    # Recover d2t absolute mapping (spec[i] -> eagle3 absolute position)
    d2t = recover_d2t(lm_spec, lm_eagle3)

    # Load checkpoint
    ckpt_path = os.path.join(checkpoint_dir, "model.safetensors")
    with safe_open(ckpt_path, framework="pt") as f:
        raw = {k: f.get_tensor(k) for k in f.keys()}

    lm_ft = raw["lm_head.weight"]
    print(f"Fine-tuned lm_head: {lm_ft.shape} {lm_ft.dtype}")

    # Check if training used identity d2t (all-zero offsets)
    if "d2t" in raw:
        ckpt_d2t = raw["d2t"]
        if (ckpt_d2t == 0).all():
            print("WARNING: checkpoint d2t is all-zeros (identity mapping). Training used")
            print("  identity d2t (spec token i -> verifier token i). Using identity expansion.")
            d2t = torch.arange(lm_ft.shape[0], dtype=torch.long)
        else:
            print(f"Using recovered d2t (checkpoint d2t was non-trivial, {(ckpt_d2t != 0).sum()} non-zero)")

    # Expand: start from pre-trained Eagle3 lm_head, scatter fine-tuned weights at d2t positions
    lm_full = lm_eagle3.clone().to(torch.float16)
    lm_full[d2t] = lm_ft.to(torch.float16)
    nonzero = (lm_full.float().abs().sum(1) > 0).sum().item()
    print(f"Expanded lm_head: {lm_full.shape}, nonzero rows: {nonzero}")

    # Build output tensors (rename layers.0.X -> midlayer.X, drop d2t/t2d)
    os.makedirs(output_dir, exist_ok=True)
    tensors = {}
    for key, val in raw.items():
        if key in ("d2t", "t2d", "lm_head.weight"):
            continue
        new_key = key.replace("layers.0.", "midlayer.", 1) if key.startswith("layers.0.") else key
        tensors[new_key] = val.to(torch.float16)
    tensors["lm_head.weight"] = lm_full

    # Save weights and config
    save_file(tensors, os.path.join(output_dir, "model.safetensors"))
    shutil.copy(
        os.path.join(vllm_model_dir, "config.json"),
        os.path.join(output_dir, "config.json"),
    )
    print(f"Saved {len(tensors)} tensors to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to speculators-dev training checkpoint directory")
    parser.add_argument("--speculators-weights", required=True, help="Original pretrain safetensors used during training (for d2t recovery)")
    parser.add_argument("--vllm-model", required=True, help="Pre-trained Eagle3 vLLM model directory (NOT the verifier; must be the Eagle3 speculative model in vLLM format)")
    parser.add_argument("--output", required=True, help="Output directory for vLLM-compatible fine-tuned model")
    args = parser.parse_args()

    convert(args.checkpoint, args.speculators_weights, args.vllm_model, args.output)


if __name__ == "__main__":
    main()
