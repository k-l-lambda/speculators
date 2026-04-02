#!/usr/bin/env python3
"""EP=16 DP=16 baseline throughput benchmark (v2).
c=32,64,128 with 500 prompts. Each concurrency level gets a warmup round first.
"""
import asyncio, aiohttp, json, time, urllib.request

PORT = 8200
API = f'http://127.0.0.1:{PORT}/v1/chat/completions'
data_500 = json.load(open('/tmp/bench_prompts_500_sharegpt.json'))
warmup_data = data_500[:20]
model_id = json.loads(
    urllib.request.urlopen(f'http://127.0.0.1:{PORT}/v1/models').read()
)['data'][0]['id']
print(f'Model: {model_id}', flush=True)

async def send(session, sem, p):
    async with sem:
        msgs = [
            {
                'role': 'user' if m['from'] == 'human' else 'assistant',
                'content': m['value'],
            }
            for m in p['conversations']
            if m['from'] in ('human',)
        ][-1:]
        async with session.post(
            API,
            json={
                'model': model_id,
                'messages': msgs,
                'max_tokens': min(p.get('output_len', 200), 2048),
                'temperature': 0,
            },
        ) as r:
            d = await r.json()
            return d.get('usage', {}).get('completion_tokens', 0)

async def run_batch(c, prompts, label=''):
    n = len(prompts)
    sem = asyncio.Semaphore(c)
    t0 = time.time()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600)
    ) as s:
        results = await asyncio.gather(*[send(s, sem, p) for p in prompts])
    elapsed = time.time() - t0
    total = sum(results)
    ok = sum(1 for r in results if r > 0)
    print(
        f'{label}c={c:3d}  prompts={n}  tok/s={total/elapsed:.1f}  '
        f'per-user={total/elapsed/c:.1f}  '
        f'ok={ok}/{n}  elapsed={elapsed:.1f}s',
        flush=True,
    )

async def main():
    for c in [32, 64, 128]:
        # Warmup: 20 prompts at same concurrency
        print(f'--- Warming up c={c} (20 prompts) ---', flush=True)
        await run_batch(c, warmup_data, label='[warmup] ')
        # Benchmark: 500 prompts
        print(f'--- Benchmark c={c} (500 prompts) ---', flush=True)
        await run_batch(c, data_500, label='[bench]  ')

asyncio.run(main())
