#!/usr/bin/env python3
"""EP=16 DP=16 baseline throughput benchmark.
c=1,2,4,8,16 use 200 prompts; c=32,64,128 use 500 prompts.
Run each concurrency level sequentially.
"""
import asyncio, aiohttp, json, time, urllib.request, sys

PORT = 8200
API = f'http://127.0.0.1:{PORT}/v1/chat/completions'
data_500 = json.load(open('/tmp/bench_prompts_500_sharegpt.json'))
data_200 = data_500[:200]
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

async def bench(c, prompts):
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
        f'c={c:3d}  prompts={n}  tok/s={total/elapsed:.1f}  '
        f'per-user={total/elapsed/c:.1f}  '
        f'ok={ok}/{n}  elapsed={elapsed:.1f}s',
        flush=True,
    )

async def main():
    # c=1,2,4,8,16 with 200 prompts
    for c in [1, 2, 4, 8, 16]:
        await bench(c, data_200)
    # c=32,64,128 with 500 prompts
    for c in [32, 64, 128]:
        await bench(c, data_500)

asyncio.run(main())
