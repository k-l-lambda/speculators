#!/usr/bin/env python3
"""Convert ZClawBench Anthropic-format trajectories to K2.5 chat template compatible messages.
Then tokenize and save as .pt files with {input_ids, loss_mask}.

Conversion rules:
- user turn with text blocks -> user message (content = concatenated text)
- user turn with tool_result blocks -> tool role messages (one per tool_result)
- assistant turn with thinking blocks -> reasoning_content field
- assistant turn with text blocks -> content field
- assistant turn with tool_use blocks -> tool_calls array

loss_mask: 0 for user/tool/system tokens, 1 for assistant tokens.
"""
import json, os, sys, torch

# Need proxy for HF dataset download
os.environ.setdefault('https_proxy', 'http://localhost:1081')
os.environ.setdefault('http_proxy', 'http://localhost:1081')

from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_PATH = '/data/models/Kimi-K2.5'
OUTPUT_DIR = '/data/datasets/zclawbench/generate'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('Loading tokenizer...', flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print('Loading ZClawBench...', flush=True)
ds = load_dataset('zai-org/ZClawBench', split='train')
print(f'Loaded {len(ds)} samples', flush=True)


def convert_trajectory(traj):
    """Convert Anthropic-format trajectory to K2.5 compatible messages list."""
    messages = []
    for turn in traj:
        role = turn['role']
        content = turn['content']

        if role == 'user':
            if isinstance(content, list):
                has_tool_result = any(c.get('type') == 'tool_result' for c in content)
                if has_tool_result:
                    for c in content:
                        if c['type'] == 'tool_result':
                            tool_content = c.get('content', '')
                            if isinstance(tool_content, list):
                                tool_content = '\n'.join(
                                    item.get('text', str(item))
                                    for item in tool_content
                                    if isinstance(item, dict)
                                )
                            messages.append({
                                'role': 'tool',
                                'content': str(tool_content)[:4000],
                                'tool_call_id': c.get('tool_use_id', 'unknown'),
                            })
                        elif c['type'] == 'text':
                            text = c.get('text', '').strip()
                            if text:
                                messages.append({
                                    'role': 'user',
                                    'content': text,
                                })
                else:
                    texts = [c.get('text', '') for c in content if c.get('type') == 'text']
                    text = '\n'.join(texts).strip()
                    if text:
                        messages.append({
                            'role': 'user',
                            'content': text,
                        })
            elif isinstance(content, str):
                if content.strip():
                    messages.append({
                        'role': 'user',
                        'content': content.strip(),
                    })

        elif role == 'assistant':
            msg = {'role': 'assistant'}
            thinking_parts = []
            text_parts = []
            tool_calls = []

            if isinstance(content, list):
                for c in content:
                    ctype = c.get('type', '')
                    if ctype == 'thinking':
                        thinking_parts.append(c.get('thinking', ''))
                    elif ctype == 'text':
                        text_parts.append(c.get('text', ''))
                    elif ctype == 'tool_use':
                        tool_calls.append({
                            'id': c.get('id', 'call_unknown'),
                            'type': 'function',
                            'function': {
                                'name': c.get('name', 'unknown'),
                                'arguments': json.dumps(c.get('input', {}), ensure_ascii=False),
                            }
                        })
            elif isinstance(content, str):
                text_parts.append(content)

            if thinking_parts:
                msg['reasoning_content'] = '\n'.join(thinking_parts)
            msg['content'] = '\n'.join(text_parts).strip() or None
            if tool_calls:
                msg['tool_calls'] = tool_calls

            messages.append(msg)

    return messages


def tokenize_with_mask(messages):
    """Tokenize messages and create loss_mask (1 for assistant tokens, 0 otherwise).
    Strategy: tokenize incrementally to find assistant token boundaries.
    """
    assistant_ranges = []
    prev_len = 0
    partial_messages = []

    for msg in messages:
        partial_messages.append(msg)
        try:
            text = tok.apply_chat_template(
                partial_messages, tokenize=False, add_generation_prompt=False
            )
            ids = tok.encode(text)
            curr_len = len(ids)
        except Exception:
            partial_messages.pop()
            continue

        if msg['role'] == 'assistant':
            assistant_ranges.append((prev_len, curr_len))

        prev_len = curr_len

    if not partial_messages:
        return None, None

    full_text = tok.apply_chat_template(
        partial_messages, tokenize=False, add_generation_prompt=False
    )
    input_ids = torch.tensor(tok.encode(full_text), dtype=torch.long)

    loss_mask = torch.zeros(len(input_ids), dtype=torch.long)
    for start, end in assistant_ranges:
        loss_mask[start:end] = 1

    return input_ids, loss_mask


# Process all samples
success = 0
skipped = 0
errors = []

for i, row in enumerate(ds):
    out_path = os.path.join(OUTPUT_DIR, f'data_{i:05d}.pt')
    if os.path.exists(out_path):
        success += 1
        continue

    try:
        traj = json.loads(row['trajectory'])
        messages = convert_trajectory(traj)

        if len(messages) < 2:
            skipped += 1
            continue

        input_ids, loss_mask = tokenize_with_mask(messages)
        if input_ids is None or len(input_ids) < 10:
            skipped += 1
            continue

        if loss_mask.sum() < 5:
            skipped += 1
            continue

        torch.save({
            'input_ids': input_ids,
            'loss_mask': loss_mask,
        }, out_path)
        success += 1

    except Exception as e:
        errors.append((i, str(e)))
        if len(errors) <= 5:
            print(f'  Error sample {i}: {e}', flush=True)

    if (i + 1) % 100 == 0:
        print(f'  Progress: {i+1}/{len(ds)}, success={success}, skipped={skipped}, errors={len(errors)}', flush=True)

print(f'\nDone: {success} saved, {skipped} skipped, {len(errors)} errors out of {len(ds)}', flush=True)

# Verify a sample
if success > 0:
    for j in range(len(ds)):
        p = os.path.join(OUTPUT_DIR, f'data_{j:05d}.pt')
        if os.path.exists(p):
            sample = torch.load(p)
            print(f'\nSample {j}: input_ids={sample["input_ids"].shape}, '
                  f'loss_mask sum={sample["loss_mask"].sum().item()}, '
                  f'total={len(sample["input_ids"])}')
            print(f'  Assistant token ratio: {sample["loss_mask"].float().mean():.2%}')
            break
