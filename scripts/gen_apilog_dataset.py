#!/usr/bin/env python3
"""
Generate Eagle3/MTP training data from K2.5 API log CSV dataset.

Two modes per row:
  - WITH response (output_tokens non-empty): tokenize full conversation -> prefill-only forward
  - WITHOUT response: K2.5 autoregressive generation -> then prefill forward

Output: .pt files in Eagle3 4-layer format:
  {input_ids, hidden_states: list of 4 tensors [S, 7168], loss_mask}

Usage:
    python scripts/gen_apilog_dataset.py \\
        --csv-path /data/datasets/.../export-*.csv \\
        --model-path /data/.cache_claude/.../Kimi-K2.5/snapshots/... \\
        --train-output /data/apilog_k25_eagle3/train/ \\
        --val-output   /data/apilog_k25_eagle3/val/ \\
        --train-size 8000 --val-size 1000 \\
        --seq-length 4096 --layer-ids 2 30 58 60 \\
        --tensor-parallel-size 8 --max-gen-tokens 512
"""

import argparse
import collections
import csv
import gc
import io
import json
import logging
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from tqdm import tqdm

os.environ["TRUST_REMOTE_CODE"] = "1"
csv.field_size_limit(sys.maxsize)

from vllm import envs
envs.VLLM_WORKER_MULTIPROC_METHOD = "spawn"

from transformers import AutoTokenizer
from speculators.data_generation.preprocessing import _normalize_conversation  # noqa: F401
from speculators.data_generation.vllm_hidden_states_generator import VllmHiddenStatesGenerator

# Import tokenize_with_loss_mask from mtp_eval_generate (handles K2.5 tiktoken)
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _scripts_dir)
from mtp_eval_generate import tokenize_with_loss_mask  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MAX_IO_WORKERS = 4


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

class NullStrippingWrapper(io.TextIOBase):
    """Strip NUL bytes present around row 100K in the archive CSV."""
    def __init__(self, stream):
        self._stream = stream

    def readable(self):
        return True

    def readline(self):
        line = self._stream.readline()
        return line.replace('\x00', '') if line else line


def load_rows_from_csv(csv_path: str) -> list:
    """Load ChatCompletionStream rows from extracted CSV.

    Returns list of dicts: {'messages': [...], 'response': str}.
    """
    rows = []
    parse_errors = 0
    total = 0

    log.info(f"Loading CSV: {csv_path}")
    with open(csv_path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(NullStrippingWrapper(f))
        for row in reader:
            total += 1
            if total % 50000 == 0:
                log.info(f"  ... {total:,} rows scanned, {len(rows):,} kept")

            if row.get('request_type', '').strip() != 'ChatCompletionStream':
                continue
            raw_body = row.get('request_body', '')
            if not raw_body:
                continue
            try:
                body = json.loads(raw_body)
            except Exception:
                parse_errors += 1
                continue
            messages = body.get('messages', [])
            if not messages:
                continue
            response = (row.get('output_tokens', '') or '').strip()
            rows.append({'messages': messages, 'response': response})

    with_resp = sum(1 for r in rows if r['response'])
    log.info(f"CSV scan complete: {total:,} total, {len(rows):,} ChatCompletionStream kept")
    log.info(f"  With response: {with_resp:,} ({with_resp / max(len(rows), 1) * 100:.1f}%)")
    log.info(f"  Without response: {len(rows) - with_resp:,}")
    log.info(f"  Parse errors: {parse_errors:,}")
    return rows


# ---------------------------------------------------------------------------
# Response generation (Phase 3)
# ---------------------------------------------------------------------------

def generate_missing_responses(
    all_rows: list,
    tokenizer,
    model_path: str,
    tp_size: int,
    seq_length: int,
    max_gen_tokens: int,
) -> list:
    """For rows without response: run K2.5 autoregressive generation.

    Items whose tokenized input fills >= seq_length - 10 tokens are skipped
    (no room to generate anything useful). Destroys the vLLM LLM instance
    after generation to free GPU memory before loading VllmHiddenStatesGenerator.
    """
    need_gen = [(i, row) for i, row in enumerate(all_rows) if not row['response']]
    if not need_gen:
        log.info("No items need generation - all have existing responses.")
        return all_rows

    log.info(f"Generating responses for {len(need_gen)} items without response...")

    need_gen_filtered = []
    for i, row in need_gen:
        try:
            prefix_ids = tokenizer.apply_chat_template(
                row['messages'], tokenize=True, add_generation_prompt=True,
            )
        except Exception as e:
            log.warning(f"  Row {i}: apply_chat_template error: {e}")
            continue
        if len(prefix_ids) < seq_length - 10:
            need_gen_filtered.append((i, row, prefix_ids))
        # else: input too long, skip generation for this item

    log.info(f"  After filtering long inputs: {len(need_gen_filtered)} items to generate")
    if not need_gen_filtered:
        return all_rows

    from vllm import TokensPrompt
    prompts = [TokensPrompt(prompt_token_ids=ids) for _, _, ids in need_gen_filtered]





    from vllm import LLM, SamplingParams

    log.info("  Initializing vLLM LLM for generation...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        max_model_len=seq_length + max_gen_tokens,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
    )
    sampling = SamplingParams(max_tokens=max_gen_tokens, temperature=0.7, top_p=0.9)
    log.info(f"  Generating {len(prompts)} responses (max_tokens={max_gen_tokens})...")
    outputs = llm.generate(prompts, sampling)

    for (i, row, _), out in zip(need_gen_filtered, outputs):
        all_rows[i]['response'] = out.outputs[0].text

    generated_count = sum(1 for r in all_rows if r['response'])
    log.info(f"  Generation complete. Items with response: {generated_count}/{len(all_rows)}")

    # CRITICAL: destroy LLM to free GPU memory before loading VllmHiddenStatesGenerator
    log.info("  Destroying vLLM LLM to free GPU memory...")
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    log.info("  GPU memory freed.")

    return all_rows


# ---------------------------------------------------------------------------
# Hidden state capture (Phase 5+6)
# ---------------------------------------------------------------------------

def run_hidden_state_capture(
    rows: list,
    tokenizer,
    model_path: str,
    layer_ids: list,
    tp_size: int,
    seq_length: int,
    output_dir: str,
    batch_size: int = 8,
):
    """Tokenize conversations and capture 4-layer hidden states via prefill.

    Returns (saved_count, token_freq_counter).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build complete conversations (input messages + response as final assistant turn)
    conversations = []
    for row in rows:
        if not row.get('response'):
            continue
        msgs = list(row['messages']) + [{"role": "assistant", "content": row['response']}]
        conversations.append(msgs)

    log.info(f"Tokenizing {len(conversations)} complete conversations...")
    preprocessed = tokenize_with_loss_mask(tokenizer, conversations, seq_length)
    num_samples = len(preprocessed["input_ids"])
    log.info(f"Successfully tokenized: {num_samples} samples")

    if num_samples == 0:
        log.error("No samples tokenized - aborting hidden state capture.")
        return 0, collections.Counter()

    total_tokens = sum(len(ids) for ids in preprocessed["input_ids"])
    total_response = sum(m.sum().item() for m in preprocessed["loss_mask"])
    log.info(
        f"Token stats: {total_tokens:,} total, {total_response:,} trainable "
        f"({total_response / total_tokens * 100:.1f}%)"
    )

    # Filter out samples with no trainable tokens (empty responses)
    valid_pairs = [
        (ids, mask)
        for ids, mask in zip(preprocessed["input_ids"], preprocessed["loss_mask"])
        if mask.sum() > 0
    ]
    log.info(f"Samples with trainable tokens: {len(valid_pairs)}/{num_samples}")
    input_ids_list = [p[0] for p in valid_pairs]
    loss_masks_list = [p[1] for p in valid_pairs]

    log.info(f"Initializing VllmHiddenStatesGenerator (layer_ids={layer_ids})...")
    generator = VllmHiddenStatesGenerator(
        model_path=model_path,
        layer_ids=layer_ids,
        max_model_len=seq_length,
        tensor_parallel_size=tp_size,
    )

    log.info(f"Extracting hidden states for {len(input_ids_list)} samples...")
    saved = 0
    token_freq = collections.Counter()
    num_batches = (len(input_ids_list) + batch_size - 1) // batch_size

    with ThreadPoolExecutor(max_workers=MAX_IO_WORKERS) as executor:
        futures = []
        pbar = tqdm(
            range(0, len(input_ids_list), batch_size),
            total=num_batches,
            desc="Extracting hidden states",
        )

        for i in pbar:
            batch_end = min(i + batch_size, len(input_ids_list))
            batch_ids   = input_ids_list[i:batch_end]
            batch_masks = loss_masks_list[i:batch_end]

            results = generator.generate(batch_ids)

            for result, mask in zip(results, batch_masks):
                input_len = len(result["input_ids"])
                trimmed_mask = mask[:input_len]

                if trimmed_mask.sum() == 0:
                    continue

                # Count token frequencies in trainable positions
                for tok, m in zip(result["input_ids"].tolist(), trimmed_mask.tolist()):
                    if m == 1:
                        token_freq[tok] += 1

                data_dict = {
                    "input_ids":     result["input_ids"],
                    "hidden_states": result["hidden_states"],  # list of 4 tensors [S, 7168]
                    "loss_mask":     trimmed_mask,
                }
                out_path = os.path.join(output_dir, f"data_{saved}.pt")
                futures.append(executor.submit(torch.save, data_dict, out_path))
                saved += 1

        log.info("Waiting for file I/O to complete...")
        for future in as_completed(futures):
            future.result()

    log.info(f"Saved {saved} .pt files to {output_dir}")
    return saved, token_freq


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv-path",    required=True, help="Path to extracted CSV")
    p.add_argument("--model-path",  required=True, help="K2.5 local snapshot path")
    p.add_argument("--train-output", default="/data/apilog_k25_eagle3/train/")
    p.add_argument("--val-output",   default="/data/apilog_k25_eagle3/val/")
    p.add_argument("--train-size",  type=int, default=8000)
    p.add_argument("--val-size",    type=int, default=1000)
    p.add_argument("--seq-length",  type=int, default=4096)
    p.add_argument("--layer-ids",   type=int, nargs="+", default=[2, 30, 58, 60])
    p.add_argument("--tensor-parallel-size", type=int, default=8)
    p.add_argument("--max-gen-tokens", type=int, default=512,
                   help="Max tokens to generate for items without response")
    p.add_argument("--batch-size",  type=int, default=8,
                   help="Batch size for hidden state extraction")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    log.info("=" * 60)
    log.info("K2.5 API Log -> Eagle3 Dataset Generator")
    log.info("=" * 60)
    log.info(f"  CSV:    {args.csv_path}")
    log.info(f"  Train:  {args.train_output} ({args.train_size} samples)")
    log.info(f"  Val:    {args.val_output} ({args.val_size} samples)")
    log.info(f"  layers: {args.layer_ids}, seq: {args.seq_length}, TP: {args.tensor_parallel_size}")

    # Phase 1: Load CSV
    all_rows = load_rows_from_csv(args.csv_path)

    # Phase 2: Shuffle & split
    rng = random.Random(args.seed)
    rng.shuffle(all_rows)
    sampled = all_rows[:args.train_size + args.val_size]
    train_rows = sampled[:args.train_size]
    val_rows   = sampled[args.train_size:]
    log.info(f"Sampled: {len(train_rows)} train + {len(val_rows)} val rows")

    # Phase 3: Load tokenizer
    log.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for name, rows in [("train", train_rows), ("val", val_rows)]:
        missing = sum(1 for r in rows if not r['response'])
        log.info(f"{name}: {missing}/{len(rows)} items need generation")

    # Phase 4: Generate missing responses (train then val)
    train_rows = generate_missing_responses(
        train_rows, tokenizer, args.model_path,
        args.tensor_parallel_size, args.seq_length, args.max_gen_tokens,
    )
    val_rows = generate_missing_responses(
        val_rows, tokenizer, args.model_path,
        args.tensor_parallel_size, args.seq_length, args.max_gen_tokens,
    )

    # Phase 5+6: Capture hidden states
    log.info("=" * 40)
    log.info("TRAIN: capturing hidden states...")
    train_saved, train_freq = run_hidden_state_capture(
        train_rows, tokenizer, args.model_path, args.layer_ids,
        args.tensor_parallel_size, args.seq_length, args.train_output, args.batch_size,
    )

    log.info("=" * 40)
    log.info("VAL: capturing hidden states...")
    val_saved, val_freq = run_hidden_state_capture(
        val_rows, tokenizer, args.model_path, args.layer_ids,
        args.tensor_parallel_size, args.seq_length, args.val_output, args.batch_size,
    )

    # Phase 7: Save token frequencies + metadata
    for out_dir, freq, name, saved_n in [
        (args.train_output, train_freq, "train", train_saved),
        (args.val_output,   val_freq,   "val",   val_saved),
    ]:
        torch.save(dict(freq), os.path.join(os.path.dirname(out_dir.rstrip("/")), f"token_freq_{name}.pt"))
        meta = {
            "csv_path":    args.csv_path,
            "model_path":  args.model_path,
            "layer_ids":   args.layer_ids,
            "seq_length":  args.seq_length,
            "seed":        args.seed,
            "split":       name,
            "target_size": args.train_size if name == "train" else args.val_size,
            "saved_files": saved_n,
        }
        with open(os.path.join(os.path.dirname(out_dir.rstrip("/")), f"meta_{name}.json"), "w") as f:
            json.dump(meta, f, indent=2)

    log.info("=" * 60)
    log.info(f"DONE. Train: {train_saved} files, Val: {val_saved} files")
    log.info(f"  Train -> {args.train_output}")
    log.info(f"  Val   -> {args.val_output}")


if __name__ == "__main__":
    main()
