#!/usr/bin/env python3
"""
Phase 1: Generate K2.5 hidden states + loss masks for MTP evaluation.

Handles K2.5's tiktoken tokenizer (no offset_mapping support) by computing
loss masks via prefix-length comparison.

Usage:
    python mtp_eval_generate.py \
        --model-path /data/.cache_claude/huggingface/hub/models--moonshotai--Kimi-K2.5/snapshots/... \
        --output-dir /data/mtp_eval/ \
        --max-samples 200 \
        --seq-length 2048
"""

import argparse
import json
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from tqdm import tqdm

# Set trust_remote_code globally
os.environ["TRUST_REMOTE_CODE"] = "1"

# Set vLLM multiproc method before importing vLLM
from vllm import envs
envs.VLLM_WORKER_MULTIPROC_METHOD = "spawn"

from datasets import load_dataset
from transformers import AutoTokenizer

from speculators.data_generation.preprocessing import (
    load_raw_dataset,
    _normalize_conversation,
)
from speculators.data_generation.vllm_hidden_states_generator import (
    VllmHiddenStatesGenerator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

MAX_IO_WORKERS = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1: Generate dataset + extract hidden states for MTP eval"
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to base K2.5 model (num_nextn_predict_layers=0)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to save .pt files",
    )
    parser.add_argument(
        "--max-samples", type=int, default=200,
        help="Number of ShareGPT conversations to use",
    )
    parser.add_argument(
        "--skip", type=int, default=0,
        help="Skip first N samples after shuffle (for independent test sets)",
    )
    parser.add_argument(
        "--seq-length", type=int, default=2048,
        help="Max sequence length",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=8,
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for hidden states extraction",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--hf-cache-dir", type=str, default=None,
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--layer-id", type=int, default=60,
        help="Layer index to capture hidden states from (default: 60, last layer)",
    )
    return parser.parse_args()


def tokenize_with_loss_mask(tokenizer, conversations, max_length):
    """Tokenize conversations and create loss masks for K2.5 tiktoken tokenizer.

    Since K2.5's tokenizer doesn't support return_offset_mapping, we compute
    loss masks by comparing token counts between prefix-only and full sequences.

    For each conversation, we find assistant turn boundaries by tokenizing
    incrementally: tokens before each assistant turn are masked (0), tokens
    within assistant turns are unmasked (1).
    """
    results = {"input_ids": [], "loss_mask": []}

    for conv in tqdm(conversations, desc="Tokenizing"):
        if not conv or not isinstance(conv, list):
            continue

        normalized = _normalize_conversation(conv)
        if not normalized:
            continue

        # Get full tokenized sequence
        try:
            full_ids = tokenizer.apply_chat_template(
                normalized,
                tokenize=True,
                add_generation_prompt=False,
            )
        except Exception as e:
            log.warning(f"Failed to tokenize conversation: {e}")
            continue

        if not full_ids or len(full_ids) < 10:
            continue

        # Truncate to max_length
        full_ids = full_ids[:max_length]

        # Build loss mask by finding assistant turn boundaries
        # Strategy: tokenize prefix up to each assistant turn start, mark the rest
        loss_mask = torch.zeros(len(full_ids), dtype=torch.long)

        # Build incremental prefixes ending before/after each assistant turn
        prefix_convs = []
        for i, turn in enumerate(normalized):
            prefix_convs.append(turn)
            if turn["role"] == "assistant":
                # Tokenize prefix up to (but not including) this assistant turn
                prefix_before = normalized[:i]
                if prefix_before:
                    try:
                        prefix_ids = tokenizer.apply_chat_template(
                            prefix_before,
                            tokenize=True,
                            add_generation_prompt=True,
                        )
                        prefix_len = min(len(prefix_ids), len(full_ids))
                    except Exception:
                        prefix_len = 0
                else:
                    prefix_len = 0

                # Tokenize prefix including this assistant turn
                prefix_with_asst = normalized[:i + 1]
                try:
                    full_prefix_ids = tokenizer.apply_chat_template(
                        prefix_with_asst,
                        tokenize=True,
                        add_generation_prompt=False,
                    )
                    asst_end = min(len(full_prefix_ids), len(full_ids))
                except Exception:
                    asst_end = len(full_ids)

                # Mark assistant tokens
                loss_mask[prefix_len:asst_end] = 1

        results["input_ids"].append(torch.tensor(full_ids, dtype=torch.long))
        results["loss_mask"].append(loss_mask)

    return results


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Phase 1: Dataset Generation + Hidden State Extraction")
    log.info("=" * 60)

    # Step 1: Load tokenizer
    log.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 2: Load ShareGPT dataset
    log.info("Loading ShareGPT dataset...")
    raw_dataset = load_raw_dataset("sharegpt", cache_dir=args.hf_cache_dir)

    random.seed(args.seed)
    indices = list(range(len(raw_dataset)))
    random.shuffle(indices)
    if args.skip > 0:
        indices = indices[args.skip:]
    if len(indices) > args.max_samples:
        indices = indices[:args.max_samples]
    raw_dataset = raw_dataset.select(indices)
    log.info(f"Selected {len(raw_dataset)} samples")

    # Step 3: Tokenize with custom loss mask
    log.info("Tokenizing with K2.5-compatible loss mask...")
    conversations = raw_dataset["conversations"]
    preprocessed = tokenize_with_loss_mask(tokenizer, conversations, args.seq_length)
    num_samples = len(preprocessed["input_ids"])
    log.info(f"Successfully tokenized {num_samples} samples")

    if num_samples == 0:
        log.error("No samples were successfully tokenized. Aborting.")
        return

    # Verify loss mask quality
    total_tokens = sum(len(ids) for ids in preprocessed["input_ids"])
    total_response = sum(m.sum().item() for m in preprocessed["loss_mask"])
    log.info(f"Total tokens: {total_tokens}, response tokens: {total_response} "
             f"({total_response / total_tokens * 100:.1f}%)")

    # Step 4: Extract hidden states
    log.info("Initializing VllmHiddenStatesGenerator...")
    generator = VllmHiddenStatesGenerator(
        model_path=args.model_path,
        layer_ids=[args.layer_id],
        max_model_len=args.seq_length,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    log.info("Extracting hidden states...")
    file_idx = 0
    num_batches = (num_samples + args.batch_size - 1) // args.batch_size
    pbar = tqdm(range(0, num_samples, args.batch_size),
                desc="Extracting hidden states", total=num_batches)

    sample_lengths = {}

    with ThreadPoolExecutor(max_workers=MAX_IO_WORKERS) as executor:
        futures = []

        for i in pbar:
            batch_end = min(i + args.batch_size, num_samples)
            batch_input_ids = preprocessed["input_ids"][i:batch_end]
            batch_loss_mask = preprocessed["loss_mask"][i:batch_end]

            results = generator.generate(batch_input_ids)

            for j, result in enumerate(results):
                input_len = len(result["input_ids"])
                loss_mask = batch_loss_mask[j][:input_len]
                sample_lengths[str(file_idx)] = input_len

                data_dict = {
                    "input_ids": result["input_ids"],
                    "hidden_states": result["hidden_states"][0],  # Single layer [S, D]
                    "loss_mask": loss_mask,
                }
                output_path = output_dir / f"data_{file_idx}.pt"
                future = executor.submit(torch.save, data_dict, str(output_path))
                futures.append(future)
                file_idx += 1

        log.info("Waiting for file saves to complete...")
        for future in as_completed(futures):
            future.result()

    # Save metadata
    meta = {
        "model_path": args.model_path,
        "layer_id": args.layer_id,
        "seq_length": args.seq_length,
        "num_samples": file_idx,
        "seed": args.seed,
        "sample_lengths": sample_lengths,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Saved {file_idx} samples to {output_dir}")
    log.info("Phase 1 complete!")


if __name__ == "__main__":
    main()
