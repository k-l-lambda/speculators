"""Extract prompts from val .pt files to genai-bench compatible jsonl format.

Usage:
    python scripts/extract_prompts_for_bench.py \
        --data-dir /data/datasets/apilog_k25_eagle3/val_5k/ \
        --tokenizer-path /path/to/Kimi-K2.5 \
        --output /tmp/bench_prompts.jsonl \
        --max-samples 500 \
        --min-prompt-tokens 100 \
        --max-prompt-tokens 4096
"""

import argparse
import glob
import json
import os
import random

import torch
from transformers import AutoTokenizer


def extract_chat_messages(tokenizer, input_ids, loss_mask):
    """Extract chat messages from input_ids using special tokens."""
    ids = input_ids.tolist()
    mask = loss_mask.tolist()

    # Find prompt/response boundary
    prompt_end = len(ids)
    for i, m in enumerate(mask):
        if m == 1:
            prompt_end = i
            break

    prompt_ids = ids[:prompt_end]
    resp_ids = ids[prompt_end:]

    # Decode to text
    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    resp_text = tokenizer.decode(resp_ids, skip_special_tokens=False)

    # Parse chat format: <|im_system|>system<|im_middle|>...<|im_end|>
    # <|im_assistant|>assistant<|im_middle|>...
    messages = []

    # Split by im_end to get turns
    IM_END = "<|im_end|>"
    IM_SYSTEM = "<|im_system|>"
    IM_ASSISTANT = "<|im_assistant|>"
    IM_MIDDLE = "<|im_middle|>"

    turns = prompt_text.split(IM_END)
    for turn in turns:
        turn = turn.strip()
        if not turn:
            continue
        if IM_SYSTEM in turn:
            content = turn.split(IM_MIDDLE, 1)[-1].strip()
            messages.append({"role": "system", "content": content})
        elif IM_ASSISTANT in turn:
            content = turn.split(IM_MIDDLE, 1)[-1].strip()
            messages.append({"role": "assistant", "content": content})
        elif "<|im_user|>" in turn or "user" in turn.split(IM_MIDDLE, 1)[0].lower():
            content = turn.split(IM_MIDDLE, 1)[-1].strip()
            messages.append({"role": "user", "content": content})

    return messages, len(prompt_ids), len(resp_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--min-prompt-tokens", type=int, default=100)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    files = sorted(glob.glob(os.path.join(args.data_dir, "data_*.pt")))
    print(f"Found {len(files)} files")

    random.seed(args.seed)
    random.shuffle(files)

    results = []
    skipped = {"too_short": 0, "too_long": 0, "no_messages": 0}

    for f in files:
        if len(results) >= args.max_samples:
            break

        d = torch.load(f, map_location="cpu")
        ids = d["input_ids"]
        mask = d["loss_mask"]

        prompt_len = (mask == 0).sum().item()
        resp_len = (mask == 1).sum().item()

        if prompt_len < args.min_prompt_tokens:
            skipped["too_short"] += 1
            continue
        if prompt_len > args.max_prompt_tokens:
            skipped["too_long"] += 1
            continue

        messages, _, _ = extract_chat_messages(tokenizer, ids, mask)
        if not messages or not any(m["role"] == "user" for m in messages):
            skipped["no_messages"] += 1
            continue

        results.append({
            "messages": messages,
            "prompt_tokens": prompt_len,
            "response_tokens": resp_len,
            "file": os.path.basename(f),
        })

    # Write jsonl
    with open(args.output, "w") as out:
        for r in results:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Extracted {len(results)} prompts to {args.output}")
    print(f"Skipped: {skipped}")

    # Stats
    prompt_lens = [r["prompt_tokens"] for r in results]
    resp_lens = [r["response_tokens"] for r in results]
    print(f"Prompt tokens: min={min(prompt_lens)}, max={max(prompt_lens)}, "
          f"mean={sum(prompt_lens)/len(prompt_lens):.0f}")
    print(f"Response tokens: min={min(resp_lens)}, max={max(resp_lens)}, "
          f"mean={sum(resp_lens)/len(resp_lens):.0f}")


if __name__ == "__main__":
    main()
