#!/usr/bin/env python3
"""Generate greedy responses for ZClawBench prompts via vLLM API.
Produces .pt files with {input_ids, loss_mask} for hidden state extraction.
"""
import asyncio, aiohttp, json, os, sys, time
import torch

API = 'http://127.0.0.1:8200/v1/chat/completions'
PROMPTS_FILE = '/data/datasets/zclawbench/prompts.jsonl'
OUTPUT_DIR = '/data/datasets/zclawbench/generate'
MAX_RESPONSE_TOKENS = 500
CONCURRENCY = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/models/Kimi-K2.5', trust_remote_code=True)

# Load prompts
prompts = []
with open(PROMPTS_FILE) as f:
    for line in f:
        prompts.append(json.loads(line))
print(f'Loaded {len(prompts)} prompts', flush=True)

# Skip already generated
existing = set(f for f in os.listdir(OUTPUT_DIR) if f.endswith('.pt'))
to_generate = [(i, p) for i, p in enumerate(prompts)
                if f'data_{i:05d}.pt' not in existing]
print(f'To generate: {len(to_generate)} (skipping {len(prompts)-len(to_generate)} existing)', flush=True)

model_id = None

async def warmup(session):
    """Send a single warmup request to trigger CUDAGraph capture."""
    global model_id
    try:
        async with session.get('http://127.0.0.1:8200/v1/models') as r:
            d = await r.json()
            model_id = d['data'][0]['id']
            print(f'Model: {model_id}', flush=True)

        print('Warmup request...', flush=True)
        async with session.post(API, json={
            'model': model_id,
            'messages': [{'role': 'user', 'content': 'Hello'}],
            'max_tokens': 10,
            'temperature': 0,
        }) as r:
            d = await r.json()
            print(f'Warmup done: {d.get("choices", [{}])[0].get("message", {}).get("content", "")[:50]}', flush=True)
    except Exception as e:
        print(f'Warmup error: {e}', flush=True)

async def generate_one(session, sem, idx, prompt):
    async with sem:
        try:
            async with session.post(API, json={
                'model': model_id,
                'messages': prompt['messages'],
                'max_tokens': MAX_RESPONSE_TOKENS,
                'temperature': 0,
            }) as r:
                d = await r.json()
                if 'error' in d:
                    print(f'  [{idx}] API error: {d["error"]}', flush=True)
                    return None
                response_text = d['choices'][0]['message']['content']
        except Exception as e:
            print(f'  [{idx}] Request error: {e}', flush=True)
            return None

    # Tokenize full conversation
    messages = prompt['messages'] + [{'role': 'assistant', 'content': response_text}]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(prompt['messages'], tokenize=False, add_generation_prompt=True)

    full_ids = tokenizer.encode(full_text)
    prompt_ids = tokenizer.encode(prompt_text)

    input_ids = torch.tensor(full_ids, dtype=torch.long)
    loss_mask = torch.zeros(len(full_ids), dtype=torch.long)
    loss_mask[len(prompt_ids):] = 1  # Only response tokens

    torch.save({
        'input_ids': input_ids,
        'loss_mask': loss_mask,
    }, os.path.join(OUTPUT_DIR, f'data_{idx:05d}.pt'))
    return len(full_ids)

async def main():
    sem = asyncio.Semaphore(CONCURRENCY)
    t0 = time.time()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        await warmup(session)
        tasks = [generate_one(session, sem, i, p) for i, p in to_generate]
        results = await asyncio.gather(*tasks)

    ok = sum(1 for r in results if r is not None)
    elapsed = time.time() - t0
    print(f'Done: {ok}/{len(to_generate)} succeeded, {elapsed:.1f}s', flush=True)

asyncio.run(main())
