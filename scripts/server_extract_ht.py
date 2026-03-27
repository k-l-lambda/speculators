#!/usr/bin/env python3
"""
Extract hidden states via vLLM server (enforce_eager=False).
Sends full sequence (prompt + greedy response) as completion prompt.
Captures h_t from deepseek_mtp.py's forward hook via VLLM_DUMP_MTP_HT.

Usage:
  # Start vLLM with MTP + dump patch:
  VLLM_DUMP_MTP_HT=/tmp/ht_dump .venv/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model /data/models/Kimi-K2.5-MTP --tensor-parallel-size 8 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --host 0.0.0.0 --port 8200

  # Run this script:
  python3 server_extract_ht.py --data-dir /data/datasets/.../val_5k_v2 \
    --output-dir /data/datasets/.../val_5k_v3 --dump-dir /tmp/ht_dump
"""
import os, sys, json, glob, asyncio, time, shutil, argparse
import aiohttp, torch
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

MODEL = "/data/models/Kimi-K2.5-MTP"
API_BASE = "http://localhost:8200"
CONCURRENCY = 32  # lower than token gen to avoid dump file collisions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="Dir with data_*.pt files (source)")
    p.add_argument("--output-dir", required=True, help="Dir to write extracted .pt files")
    p.add_argument("--dump-dir", default="/tmp/ht_server_dump", help="VLLM_DUMP_MTP_HT dir")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--resume", action="store_true", default=True)
    return p.parse_args()


async def extract_one(session, sem, pt_path, greedy_json_path, dump_dir, out_path, idx):
    """Send full sequence to server, wait for h_t dump, save .pt file."""
    async with sem:
        # Load sample
        d = torch.load(pt_path, map_location="cpu", weights_only=True)
        
        # Get greedy response tokens
        with open(greedy_json_path) as f:
            greedy = json.load(f)
        
        prompt_ids = greedy["prompt_ids"]
        gen_text = greedy["gen_text"]
        
        # Encode response to get token ids
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        resp_ids = tokenizer.encode(gen_text, add_special_tokens=False)
        
        # Full sequence = prompt + response
        full_ids = prompt_ids + resp_ids
        if len(full_ids) > 4094:
            full_ids = full_ids[:4094]
        
        # Clear dump dir slot for this request
        req_id = f"req_{idx:06d}"
        
        # Send full sequence, request 1 token output (triggers MTP forward)
        try:
            async with session.post(f"{API_BASE}/v1/completions", json={
                "model": MODEL,
                "prompt": full_ids,
                "max_tokens": 1,
                "temperature": 0,
            }) as resp:
                if resp.status != 200:
                    return False
                result = await resp.json()
        except Exception as e:
            print(f"  Error {idx}: {e}")
            return False
        
        # Find the newest h_t dump file (created by this request)
        # We need request-level isolation for dump files
        # The dump patch increments a counter; we trust sequential processing
        return True


async def main_sequential(args):
    """Process sequentially to avoid dump file collisions."""
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    dump_dir = Path(args.dump_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find samples
    greedy_jsons = sorted(data_dir.glob("*_greedy.json"))
    if args.max_samples:
        greedy_jsons = greedy_jsons[:args.max_samples]
    
    # Skip already extracted
    if args.resume:
        existing = set(f.name for f in out_dir.glob("data_*.pt"))
        greedy_jsons = [f for f in greedy_jsons
                       if f.name.replace("_greedy.json", ".pt") not in existing]
    
    print(f"Extracting {len(greedy_jsons)} samples (sequential, 1 at a time)")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
        for i, greedy_json in enumerate(tqdm(greedy_jsons)):
            pt_name = greedy_json.name.replace("_greedy.json", ".pt")
            pt_path = data_dir / pt_name
            out_path = out_dir / pt_name
            
            if not pt_path.exists():
                continue
            
            with open(greedy_json) as f:
                greedy = json.load(f)
            
            prompt_ids = greedy["prompt_ids"]
            gen_text = greedy["gen_text"]
            resp_ids = tokenizer.encode(gen_text, add_special_tokens=False)
            full_ids = prompt_ids + resp_ids
            
            if len(full_ids) > 4094:
                full_ids = full_ids[:4094]
            
            # Clear dump dir
            for f in dump_dir.glob("ht_*.pt"):
                f.unlink()
            
            # Request
            try:
                async with session.post(f"{API_BASE}/v1/completions", json={
                    "model": MODEL,
                    "prompt": full_ids,
                    "max_tokens": 1,
                    "temperature": 0,
                }) as resp:
                    if resp.status != 200:
                        continue
                    await resp.json()
            except Exception as e:
                continue
            
            # Wait for dump file (with timeout)
            await asyncio.sleep(0.1)
            dump_files = sorted(dump_dir.glob("ht_*.pt"),
                               key=lambda f: f.stat().st_mtime, reverse=True)
            # Skip warmup dumps (zeros)
            ht_tensor = None
            for df in dump_files:
                d2 = torch.load(df, map_location="cpu", weights_only=True)
                if d2['h_t'].float().abs().sum() > 1.0:
                    ht_tensor = d2['h_t']  # [seq_len, 7168]
                    break
            
            if ht_tensor is None:
                continue
            
            # Load original for loss_mask
            orig = torch.load(pt_path, map_location="cpu", weights_only=True)
            lm = orig['loss_mask']
            prompt_len = (lm == 0).sum().item()
            
            # Build output: use full_ids as input_ids, h_t from server
            n = min(len(ht_tensor), len(full_ids))
            new_loss_mask = torch.zeros(n, dtype=torch.long)
            new_loss_mask[prompt_len:] = 1
            
            torch.save({
                'input_ids': torch.tensor(full_ids[:n], dtype=torch.long),
                'hidden_states': [ht_tensor[:n]],
                'loss_mask': new_loss_mask,
            }, out_path)
    
    print(f"Done! Extracted to {out_dir}")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main_sequential(args))
