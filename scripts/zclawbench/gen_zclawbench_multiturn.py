#!/usr/bin/env python3
"""Generate multi-turn K2.5 responses for ZClawBench.
For each trajectory: keep user/tool_result turns, regenerate assistant turns via K2.5.
Produces .pt files with {input_ids, loss_mask} for hidden state extraction.
"""
import asyncio, aiohttp, json, os, sys, time, traceback
import torch

API = 'http://127.0.0.1:8200/v1/chat/completions'
OUTPUT_DIR = '/data/datasets/zclawbench/generate_multiturn'
MAX_RESPONSE_TOKENS = 500
CONCURRENCY = 8  # concurrent samples (each sample is sequential multi-turn)

os.makedirs(OUTPUT_DIR, exist_ok=True)

from transformers import AutoTokenizer

print('Loading tokenizer...', flush=True)
tok = AutoTokenizer.from_pretrained('/models/Kimi-K2.5', trust_remote_code=True)

print('Loading ZClawBench trajectories...', flush=True)
TRAJ_FILE = '/data/datasets/zclawbench/trajectories.jsonl'
ds = []
with open(TRAJ_FILE) as f:
    for line in f:
        ds.append(json.loads(line))
print(f'Loaded {len(ds)} samples', flush=True)

model_id = None


def extract_user_content(content):
    """Extract text from user turn content (may be list or str)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for c in content:
            ctype = c.get('type', '')
            if ctype == 'text':
                parts.append(c.get('text', ''))
        return '\n'.join(parts).strip()
    return ''


def extract_tool_results(content):
    """Extract tool_result entries from user turn content."""
    results = []
    if isinstance(content, list):
        for c in content:
            if c.get('type') == 'tool_result':
                tool_content = c.get('content', '')
                if isinstance(tool_content, list):
                    tool_content = '\n'.join(
                        item.get('text', str(item))
                        for item in tool_content
                        if isinstance(item, dict)
                    )
                results.append({
                    'tool_use_id': c.get('tool_use_id', 'unknown'),
                    'content': str(tool_content)[:4000],
                })
    return results


async def generate_multiturn(session, sem, idx, row):
    """Process one trajectory: keep user/tool turns, regenerate assistant turns."""
    traj = json.loads(row['trajectory'])

    # Build conversation turn by turn
    # messages_so_far: K2.5-compatible messages sent to API
    # all_messages: complete conversation for tokenization
    messages_so_far = []
    all_messages = []

    for turn in traj:
        role = turn['role']
        content = turn['content']

        if role == 'user':
            # Check if tool_result or text
            has_tool_result = isinstance(content, list) and any(
                c.get('type') == 'tool_result' for c in content
            )

            if has_tool_result:
                tool_results = extract_tool_results(content)
                for tr in tool_results:
                    msg = {
                        'role': 'tool',
                        'content': tr['content'],
                        'tool_call_id': tr['tool_use_id'],
                    }
                    messages_so_far.append(msg)
                    all_messages.append(msg)
                # Also check for text mixed in
                text = extract_user_content(
                    [c for c in content if c.get('type') == 'text']
                )
                if text:
                    msg = {'role': 'user', 'content': text}
                    messages_so_far.append(msg)
                    all_messages.append(msg)
            else:
                text = extract_user_content(content)
                if text:
                    msg = {'role': 'user', 'content': text}
                    messages_so_far.append(msg)
                    all_messages.append(msg)

        elif role == 'assistant':
            # Generate K2.5 response for this turn
            async with sem:
                try:
                    async with session.post(API, json={
                        'model': model_id,
                        'messages': messages_so_far,
                        'max_tokens': MAX_RESPONSE_TOKENS,
                        'temperature': 0,
                    }) as r:
                        d = await r.json()
                        if 'error' in d:
                            print(f'  [{idx}] API error at turn: {d["error"]}', flush=True)
                            return None
                        response_text = d['choices'][0]['message']['content']
                except Exception as e:
                    print(f'  [{idx}] Request error: {e}', flush=True)
                    return None

            msg = {'role': 'assistant', 'content': response_text}
            messages_so_far.append(msg)
            all_messages.append(msg)

    # Tokenize the full multi-turn conversation with loss_mask
    if not all_messages:
        return None

    # Incremental tokenization to find assistant boundaries
    assistant_ranges = []
    prev_len = 0
    partial = []

    for msg in all_messages:
        partial.append(msg)
        try:
            text = tok.apply_chat_template(
                partial, tokenize=False, add_generation_prompt=False
            )
            ids = tok.encode(text)
            curr_len = len(ids)
        except Exception:
            partial.pop()
            continue

        if msg['role'] == 'assistant':
            assistant_ranges.append((prev_len, curr_len))
        prev_len = curr_len

    if not partial or not assistant_ranges:
        return None

    full_text = tok.apply_chat_template(
        partial, tokenize=False, add_generation_prompt=False
    )
    input_ids = torch.tensor(tok.encode(full_text), dtype=torch.long)
    loss_mask = torch.zeros(len(input_ids), dtype=torch.long)
    for start, end in assistant_ranges:
        loss_mask[start:end] = 1

    torch.save({
        'input_ids': input_ids,
        'loss_mask': loss_mask,
    }, os.path.join(OUTPUT_DIR, f'data_{idx:05d}.pt'))

    return {
        'tokens': len(input_ids),
        'assistant_tokens': loss_mask.sum().item(),
        'turns': len(all_messages),
        'assistant_turns': len(assistant_ranges),
    }


async def main():
    global model_id

    sem = asyncio.Semaphore(CONCURRENCY)

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600)
    ) as session:
        # Get model ID
        async with session.get('http://127.0.0.1:8200/v1/models') as r:
            d = await r.json()
            model_id = d['data'][0]['id']
        print(f'Model: {model_id}', flush=True)

        # Warmup
        print('Warmup...', flush=True)
        async with session.post(API, json={
            'model': model_id,
            'messages': [{'role': 'user', 'content': 'Hello'}],
            'max_tokens': 10,
            'temperature': 0,
        }) as r:
            await r.json()
        print('Warmup done', flush=True)

        # Skip already generated
        existing = set(f for f in os.listdir(OUTPUT_DIR) if f.endswith('.pt'))
        to_generate = [(i, ds[i]) for i in range(len(ds))
                       if f'data_{i:05d}.pt' not in existing]
        print(f'To generate: {len(to_generate)} (skipping {len(ds)-len(to_generate)} existing)',
              flush=True)

        t0 = time.time()
        tasks = [generate_multiturn(session, sem, i, row) for i, row in to_generate]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - t0

    ok = [r for r in results if r is not None]
    print(f'\nDone: {len(ok)}/{len(to_generate)} succeeded, {elapsed:.1f}s', flush=True)

    if ok:
        avg_tokens = sum(r['tokens'] for r in ok) / len(ok)
        avg_asst = sum(r['assistant_tokens'] for r in ok) / len(ok)
        avg_turns = sum(r['assistant_turns'] for r in ok) / len(ok)
        print(f'Avg tokens: {avg_tokens:.0f}, avg assistant tokens: {avg_asst:.0f}, '
              f'avg assistant turns: {avg_turns:.1f}', flush=True)


asyncio.run(main())
