#!/usr/bin/env python3
"""Generate multi-turn ZClawBench samples with truncation instead of dropping.

This version keeps all trajectories by:
1. trimming the oldest request context before each API call so prompt+response fits
2. left-truncating the final tokenized sample to MAX_MODEL_LEN tokens
"""
import asyncio
import json
import os
import time

import aiohttp
import torch
from transformers import AutoTokenizer

API = "http://127.0.0.1:8200/v1/chat/completions"
OUTPUT_DIR = "/data/datasets/zclawbench/generate_multiturn_8k_trunc"
MAX_MODEL_LEN = 8192
MAX_RESPONSE_TOKENS = 500
CONCURRENCY = 8
TOOL_CONTENT_CHAR_LIMIT = 4000

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained("/models/Kimi-K2.5", trust_remote_code=True)

print("Loading ZClawBench trajectories...", flush=True)
TRAJ_FILE = "/data/datasets/zclawbench/trajectories.jsonl"
ds = []
with open(TRAJ_FILE) as f:
    for line in f:
        ds.append(json.loads(line))
print(f"Loaded {len(ds)} samples", flush=True)

model_id = None


def extract_user_content(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts).strip()
    return ""


def extract_tool_results(content):
    results = []
    if isinstance(content, list):
        for item in content:
            if item.get("type") != "tool_result":
                continue
            tool_content = item.get("content", "")
            if isinstance(tool_content, list):
                tool_content = "\n".join(
                    part.get("text", str(part))
                    for part in tool_content
                    if isinstance(part, dict)
                )
            results.append(
                {
                    "tool_use_id": item.get("tool_use_id", "unknown"),
                    "content": str(tool_content)[:TOOL_CONTENT_CHAR_LIMIT],
                }
            )
    return results


def encode_messages(messages):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return tok.encode(text)


def crop_message_content(msg, target_chars):
    cropped = dict(msg)
    cropped["content"] = cropped.get("content", "")[-target_chars:]
    return cropped


def fit_messages_to_budget(messages, max_prompt_tokens):
    """Keep the newest context that fits the prompt token budget."""
    fitted = [dict(msg) for msg in messages]
    was_trimmed = False
    if not fitted:
        return fitted, was_trimmed

    while fitted:
        try:
            token_len = len(encode_messages(fitted))
        except Exception:
            token_len = max_prompt_tokens + 1
        if token_len <= max_prompt_tokens:
            return fitted, was_trimmed

        was_trimmed = True
        if len(fitted) > 1:
            fitted.pop(0)
            continue

        content = fitted[0].get("content", "")
        if not content:
            return fitted, was_trimmed
        target_chars = max(256, int(len(content) * 0.7))
        if target_chars >= len(content):
            target_chars = max(1, len(content) - 1)
        fitted[0] = crop_message_content(fitted[0], target_chars)

    return fitted, was_trimmed


def tokenize_with_ranges(messages):
    assistant_ranges = []
    partial = []
    prev_len = 0

    for msg in messages:
        partial.append(msg)
        try:
            curr_ids = encode_messages(partial)
        except Exception:
            partial.pop()
            continue
        curr_len = len(curr_ids)
        if msg["role"] == "assistant":
            assistant_ranges.append((prev_len, curr_len))
        prev_len = curr_len

    if not partial or not assistant_ranges:
        return None, None

    input_ids = torch.tensor(encode_messages(partial), dtype=torch.long)
    loss_mask = torch.zeros(len(input_ids), dtype=torch.long)
    for start, end in assistant_ranges:
        loss_mask[start:end] = 1
    return input_ids, loss_mask


def truncate_sample(input_ids, loss_mask, max_len):
    if len(input_ids) <= max_len:
        return input_ids, loss_mask, False
    start = len(input_ids) - max_len
    return input_ids[start:], loss_mask[start:], True


async def generate_multiturn(session, sem, idx, row):
    traj = json.loads(row["trajectory"])
    messages_so_far = []
    all_messages = []
    num_request_trims = 0

    for turn in traj:
        role = turn["role"]
        content = turn["content"]

        if role == "user":
            has_tool_result = isinstance(content, list) and any(
                item.get("type") == "tool_result" for item in content
            )

            if has_tool_result:
                for tool_result in extract_tool_results(content):
                    msg = {
                        "role": "tool",
                        "content": tool_result["content"],
                        "tool_call_id": tool_result["tool_use_id"],
                    }
                    messages_so_far.append(msg)
                    all_messages.append(msg)

                text = extract_user_content([item for item in content if item.get("type") == "text"])
                if text:
                    msg = {"role": "user", "content": text}
                    messages_so_far.append(msg)
                    all_messages.append(msg)
            else:
                text = extract_user_content(content)
                if text:
                    msg = {"role": "user", "content": text}
                    messages_so_far.append(msg)
                    all_messages.append(msg)

        elif role == "assistant":
            prompt_messages, trimmed = fit_messages_to_budget(
                messages_so_far, MAX_MODEL_LEN - MAX_RESPONSE_TOKENS
            )
            if trimmed:
                num_request_trims += 1

            async with sem:
                try:
                    async with session.post(
                        API,
                        json={
                            "model": model_id,
                            "messages": prompt_messages,
                            "max_tokens": MAX_RESPONSE_TOKENS,
                            "temperature": 0,
                        },
                    ) as resp:
                        data = await resp.json()
                        if "error" in data:
                            print(f"  [{idx}] API error at turn: {data['error']}", flush=True)
                            return None
                        response_text = data["choices"][0]["message"]["content"]
                except Exception as exc:
                    print(f"  [{idx}] Request error: {exc}", flush=True)
                    return None

            msg = {"role": "assistant", "content": response_text}
            messages_so_far.append(msg)
            all_messages.append(msg)

    if not all_messages:
        return None

    input_ids, loss_mask = tokenize_with_ranges(all_messages)
    if input_ids is None:
        return None

    input_ids, loss_mask, final_truncated = truncate_sample(input_ids, loss_mask, MAX_MODEL_LEN)
    if int(loss_mask.sum()) == 0:
        return None

    torch.save(
        {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
        },
        os.path.join(OUTPUT_DIR, f"data_{idx:05d}.pt"),
    )

    return {
        "tokens": len(input_ids),
        "assistant_tokens": int(loss_mask.sum().item()),
        "assistant_turns": sum(1 for msg in all_messages if msg["role"] == "assistant"),
        "request_trims": num_request_trims,
        "final_truncated": int(final_truncated),
    }


async def main():
    global model_id
    sem = asyncio.Semaphore(CONCURRENCY)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        async with session.get("http://127.0.0.1:8200/v1/models") as resp:
            data = await resp.json()
            model_id = data["data"][0]["id"]
        print(f"Model: {model_id}", flush=True)

        print("Warmup...", flush=True)
        async with session.post(
            API,
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10,
                "temperature": 0,
            },
        ) as resp:
            await resp.json()
        print("Warmup done", flush=True)

        existing = {name for name in os.listdir(OUTPUT_DIR) if name.endswith(".pt")}
        to_generate = [
            (i, ds[i]) for i in range(len(ds)) if f"data_{i:05d}.pt" not in existing
        ]
        print(
            f"To generate: {len(to_generate)} (skipping {len(ds) - len(to_generate)} existing)",
            flush=True,
        )

        t0 = time.time()
        tasks = [generate_multiturn(session, sem, i, row) for i, row in to_generate]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - t0

    ok = [item for item in results if item is not None]
    print(f"\nDone: {len(ok)}/{len(to_generate)} succeeded, {elapsed:.1f}s", flush=True)
    if not ok:
        return

    avg_tokens = sum(item["tokens"] for item in ok) / len(ok)
    avg_asst = sum(item["assistant_tokens"] for item in ok) / len(ok)
    avg_turns = sum(item["assistant_turns"] for item in ok) / len(ok)
    trim_samples = sum(1 for item in ok if item["request_trims"] > 0)
    final_trunc_samples = sum(item["final_truncated"] for item in ok)
    print(
        f"Avg tokens: {avg_tokens:.0f}, avg assistant tokens: {avg_asst:.0f}, "
        f"avg assistant turns: {avg_turns:.1f}",
        flush=True,
    )
    print(
        f"Request-trimmed samples: {trim_samples}/{len(ok)}, "
        f"final left-truncated samples: {final_trunc_samples}/{len(ok)}",
        flush=True,
    )


asyncio.run(main())
