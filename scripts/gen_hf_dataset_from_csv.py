#!/usr/bin/env python3
"""
Build HF dataset (input_ids + loss_mask) from K2.5 API log CSV export for dynamic training.

Differences from gen_apilog_dataset.py:
  - ONLY uses rows WITH output_tokens (skips generation entirely)
  - Does NOT extract hidden states (dynamic training does that on-the-fly)
  - Outputs HF dataset directly (no .pt files)
  - Supports streaming directly from tar.gz (no extraction needed)

Usage (inside pod on youyun.37):
    python3 /tmp/gen_hf_dataset_from_csv.py \\
        --tar-path /data/datasets/export-3c6b3075-afb1-44c8-a803-da5b1fe72734_1775096260.tar.gz \\
        --model-path /data/models/Kimi-K2.5 \\
        --output /data/datasets/export_3c6b3075_hf \\
        --train-size 30000 --val-size 3000 \\
        --seq-length 4096
"""

import argparse
import csv
import io
import json
import logging
import os
import random
import sys
import tarfile
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer

os.environ["TRUST_REMOTE_CODE"] = "1"
csv.field_size_limit(sys.maxsize)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers from gen_apilog_dataset.py
# ---------------------------------------------------------------------------

class NullStrippingWrapper(io.TextIOBase):
    """Strip NUL bytes that sometimes appear in the archive CSV."""
    def __init__(self, stream):
        self._stream = stream

    def readable(self):
        return True

    def readline(self):
        line = self._stream.readline()
        return line.replace('\x00', '') if line else line


def _open_csv_stream(path: str):
    """Open CSV file, supporting plain .csv or streaming first file inside .tar.gz."""
    if path.endswith('.tar.gz') or path.endswith('.tgz'):
        log.info(f"Streaming first CSV file from tar.gz: {path}")
        tf = tarfile.open(path, 'r:gz')
        # Find first .csv member
        for member in tf:
            if member.name.endswith('.csv') and member.isfile():
                log.info(f"  Found CSV: {member.name} (size ~{member.size / 1e9:.1f} GB uncompressed)")
                f = tf.extractfile(member)
                text_stream = io.TextIOWrapper(f, encoding='utf-8-sig', errors='replace')
                return text_stream, tf
        tf.close()
        raise RuntimeError(f"No CSV found inside {path}")
    elif path.endswith('.csv'):
        return open(path, encoding='utf-8-sig'), None
    else:
        raise ValueError(f"Unsupported file format: {path}. Expected .csv or .tar.gz")


def load_rows_with_response(path: str, max_scan: int = 0) -> list:
    """Load ChatCompletionStream rows that have non-empty output_tokens.

    Args:
        path: CSV file or tar.gz containing CSV.
        max_scan: Stop after scanning this many total rows (0 = scan entire file).

    Returns:
        List of dicts: {'messages': [...], 'response': str}
    """
    stream, tf = _open_csv_stream(path)
    rows = []
    total = 0
    skipped_type = 0
    skipped_no_resp = 0
    skipped_parse = 0

    try:
        reader = csv.DictReader(NullStrippingWrapper(stream))
        for row in reader:
            total += 1
            if total % 50_000 == 0:
                log.info(
                    f"  Scanned {total:,} rows — kept {len(rows):,} "
                    f"(skip_type={skipped_type:,} skip_resp={skipped_no_resp:,} "
                    f"skip_parse={skipped_parse:,})"
                )
            if max_scan > 0 and total > max_scan:
                log.info(f"  Reached max_scan={max_scan:,}, stopping.")
                break

            # Filter by request type
            if row.get('request_type', '').strip() != 'ChatCompletionStream':
                skipped_type += 1
                continue

            # Only rows with existing response (skip generation entirely)
            response = (row.get('output_tokens', '') or '').strip()
            if not response:
                skipped_no_resp += 1
                continue

            # Parse request_body JSON
            raw_body = row.get('request_body', '')
            if not raw_body:
                skipped_parse += 1
                continue
            try:
                body = json.loads(raw_body)
            except Exception:
                skipped_parse += 1
                continue
            messages = body.get('messages', [])
            if not messages:
                skipped_parse += 1
                continue

            rows.append({'messages': messages, 'response': response})
    finally:
        stream.close()
        if tf is not None:
            tf.close()

    with_resp = len(rows)
    log.info(
        f"CSV scan complete: {total:,} total rows\n"
        f"  Kept (with response): {with_resp:,}\n"
        f"  Skipped (wrong type): {skipped_type:,}\n"
        f"  Skipped (no response): {skipped_no_resp:,}\n"
        f"  Skipped (parse error): {skipped_parse:,}"
    )
    return rows


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def _fast_tokenize_row(tokenizer, row: dict, seq_length: int, min_response_tokens: int = 10):
    """Tokenize a single row (messages + response) with loss_mask.

    Fast path: only 2 apply_chat_template calls per row (full + prefix).
    The row's `response` (output_tokens) is the final assistant turn.
    All message roles (including tool/tool_result) are kept as-is for context.

    If the full sequence exceeds seq_length, left-truncation is applied:
    the prefix is trimmed from the left, preserving the response at the end.

    Returns (input_ids: list, loss_mask: list) or None on error.
    """
    messages = row['messages']
    response = row['response']

    # Skip multimodal content (list-type content fields)
    for m in messages:
        c = m.get('content', '')
        if isinstance(c, list):
            return None

    # Full conversation: all original messages + the response as final assistant turn
    full_conv = list(messages) + [{"role": "assistant", "content": response}]

    # Prefix = all messages (without the final assistant response)
    prefix = list(messages)
    if not prefix:
        return None

    try:
        full_ids = tokenizer.apply_chat_template(
            full_conv, tokenize=True, add_generation_prompt=False,
        )
        prefix_ids = tokenizer.apply_chat_template(
            prefix, tokenize=True, add_generation_prompt=True,
        )
    except Exception:
        return None

    if not full_ids or len(full_ids) < 5:
        return None

    # Derive response tokens = full_ids tokens after the prefix
    prefix_len = len(prefix_ids)
    resp_ids = full_ids[prefix_len:]

    if len(resp_ids) < min_response_tokens:
        return None  # Response too short (probably truncated or empty)

    # Apply left-truncation if total exceeds seq_length:
    # keep up to (seq_length - len(resp_ids)) prefix tokens from the RIGHT (most recent context)
    max_prefix = seq_length - len(resp_ids)
    if max_prefix <= 0:
        # Response alone exceeds seq_length — truncate response
        resp_ids = resp_ids[:seq_length - 16]  # keep at least 16 prefix tokens
        max_prefix = 16
    if prefix_len > max_prefix:
        prefix_ids_trimmed = prefix_ids[-max_prefix:]
    else:
        prefix_ids_trimmed = prefix_ids

    final_ids = prefix_ids_trimmed + resp_ids
    final_ids = final_ids[:seq_length]
    actual_prefix_len = len(final_ids) - len(resp_ids[:seq_length - len(prefix_ids_trimmed)])

    # Simpler: just compute actual prefix length in final_ids
    n_resp = min(len(resp_ids), len(final_ids))
    n_prefix = len(final_ids) - n_resp

    loss_mask = [0] * n_prefix + [1] * n_resp
    assert len(loss_mask) == len(final_ids)

    if sum(loss_mask) == 0:
        return None

    return final_ids, loss_mask


def build_samples(rows: list, tokenizer, seq_length: int) -> list:
    """Tokenize rows into (input_ids, loss_mask) samples using fast path."""
    log.info(f"Tokenizing {len(rows):,} rows (seq_length={seq_length})...")
    t0 = time.time()
    valid = []
    skipped = 0
    for i, row in enumerate(rows):
        if i % 5000 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(rows) - i) / rate if rate > 0 else 0
            log.info(f"  {i:,}/{len(rows):,} tokenized ({rate:.0f}/s, ETA {eta/60:.1f}m) — valid={len(valid):,}")
        result = _fast_tokenize_row(tokenizer, row, seq_length)
        if result is None:
            skipped += 1
            continue
        valid.append(result)
    elapsed = time.time() - t0
    log.info(
        f"Tokenization done in {elapsed:.1f}s ({len(rows)/elapsed:.0f}/s). "
        f"Valid: {len(valid):,}/{len(rows):,}, skipped: {skipped:,}"
    )
    return valid



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tar-path",    required=True,  help="Path to .tar.gz or .csv")
    p.add_argument("--model-path",  required=True,  help="K2.5 local model path")
    p.add_argument("--output",      required=True,  help="HF dataset output dir (contains /train and /val)")
    p.add_argument("--val-ratio",   type=float, default=0.1,
                   help="Fraction of valid samples for val (default: 0.1)")
    p.add_argument("--seq-length",  type=int, default=16384,
                   help="Max sequence length; left-truncation keeps response (default: 16384)")
    p.add_argument("--max-scan",    type=int, default=0, help="Max rows to scan (0=scan entire file)")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    log.info("=" * 60)
    log.info("K2.5 API Log CSV -> HF Dataset (for dynamic training)")
    log.info("=" * 60)
    log.info(f"  Input:      {args.tar_path}")
    log.info(f"  Output:     {args.output}")
    log.info(f"  Seq length: {args.seq_length}")
    log.info(f"  Val ratio:  {args.val_ratio}")
    log.info(f"  Max scan:   {'unlimited' if args.max_scan == 0 else str(args.max_scan) + ' rows'}")

    # Phase 1: Scan entire CSV, collect all rows with response
    log.info("
[Phase 1] Scanning full CSV for all rows with response...")
    rows = load_rows_with_response(args.tar_path, max_scan=args.max_scan)

    if not rows:
        log.error("No valid rows found!")
        return

    # Phase 2: Load tokenizer
    log.info(f"
[Phase 2] Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info(f"  Vocab size: {tokenizer.vocab_size:,}")

    # Phase 3: Tokenize ALL rows, collect valid samples
    log.info(f"
[Phase 3] Tokenizing all {len(rows):,} rows (seq_length={args.seq_length})...")
    all_samples = build_samples(rows, tokenizer, args.seq_length)

    if not all_samples:
        log.error("No valid samples after tokenization!")
        return

    # Phase 4: Shuffle and split into train/val
    log.info(f"
[Phase 4] Splitting {len(all_samples):,} valid samples (val_ratio={args.val_ratio})...")
    rng = random.Random(args.seed)
    rng.shuffle(all_samples)
    n_val   = max(1, int(len(all_samples) * args.val_ratio))
    n_train = len(all_samples) - n_val
    train_samples = all_samples[:n_train]
    val_samples   = all_samples[n_train:]
    log.info(f"  Train: {len(train_samples):,}, Val: {len(val_samples):,}")

    # Phase 5: Save HF datasets
    log.info("
[Phase 5] Saving HF datasets...")
    train_out = os.path.join(args.output, "train")
    val_out   = os.path.join(args.output, "val")
    from pathlib import Path
    Path(train_out).mkdir(parents=True, exist_ok=True)
    Path(val_out).mkdir(parents=True, exist_ok=True)

    def save_hf(samples, out_path, name):
        if not samples:
            log.error(f"  No {name} samples to save!")
            return 0
        input_ids  = [s[0] for s in samples]
        loss_masks = [s[1] for s in samples]
        lengths    = [len(ids) for ids in input_ids]
        trainable  = [sum(m) for m in loss_masks]
        log.info(
            f"  {name}: {len(samples):,} samples, "
            f"avg_len={sum(lengths)/len(lengths):.0f}, "
            f"avg_trainable={sum(trainable)/len(trainable):.0f} "
            f"({sum(trainable)/max(sum(lengths),1)*100:.1f}%)"
        )
        ds = Dataset.from_dict({"input_ids": input_ids, "loss_mask": loss_masks})
        ds.save_to_disk(out_path)
        log.info(f"  Saved {len(ds):,} {name} samples -> {out_path}")
        return len(ds)

    n_train_saved = save_hf(train_samples, train_out, "train")
    n_val_saved   = save_hf(val_samples,   val_out,   "val")

    log.info("
" + "=" * 60)
    log.info("DONE")
    log.info(f"  Train: {n_train_saved:,} samples -> {train_out}")
    log.info(f"  Val:   {n_val_saved:,} samples -> {val_out}")
    log.info(f"  Total disk: run 'du -sh {args.output}'")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
