#!/usr/bin/env python3
"""Generate greedy response dataset from val_5k prompts via vLLM API.

Sends each val_5k prompt to vLLM (temperature=0), collects greedy response tokens,
concatenates with prompt to form full sequence, then extracts INT4 hidden states.

Step 1: Generate greedy responses via vLLM API
Step 2: Extract hidden states via VllmHiddenStatesGenerator

Usage:
    # Step 1: with vLLM baseline running on port 8200
    python scripts/generate_greedy_dataset.py --step generate \
        --data-dir /data/datasets/apilog_k25_eagle3/val_5k \
        --output-dir /data/datasets/apilog_k25_eagle3/val_5k_greedy \
        --api-base http://localhost:8200 --max-response-tokens 500

    # Step 2: with GPUs free
    python scripts/generate_greedy_dataset.py --step extract \
        --output-dir /data/datasets/apilog_k25_eagle3/val_5k_greedy \
        --model-path /data/models/Kimi-K2.5-MTP
"""
import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["generate", "extract", "both"], default="both")
    parser.add_argument("--data-dir", type=str, default="/data/datasets/apilog_k25_eagle3/val_5k")
    parser.add_argument("--output-dir", type=str, default="/data/datasets/apilog_k25_eagle3/val_5k_greedy")
    parser.add_argument("--api-base", type=str, default="http://localhost:8200")
    parser.add_argument("--model-path", type=str, default="/data/models/Kimi-K2.5-MTP")
    parser.add_argument("--max-response-tokens", type=int, default=500)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--layer-ids", type=int, nargs="+", default=[2, 30, 58, 60])
    return parser.parse_args()


def step_generate(args):
    """Generate greedy responses via vLLM API."""
    import torch
    import requests
    from tqdm import tqdm

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(data_dir.glob("data_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if args.max_samples:
        pt_files = pt_files[:args.max_samples]

    print(f"Generating greedy responses for {len(pt_files)} samples")

    for pt_file in tqdm(pt_files, desc="Generating"):
        out_path = output_dir / pt_file.name.replace(".pt", "_greedy.json")
        if out_path.exists():
            continue

        sample = torch.load(str(pt_file), map_location="cpu", weights_only=True)
        ids = sample["input_ids"]
        lm = sample["loss_mask"]

        # Find prompt/response split
        rs = next((i for i in range(len(lm)) if lm[i] == 1), None)
        if rs is None or rs < 5:
            continue

        prompt_ids = ids[:rs].tolist()
        orig_resp_len = len(ids) - rs
        max_tokens = min(orig_resp_len, args.max_response_tokens)

        try:
            resp = requests.post(f"{args.api_base}/v1/completions", json={
                "model": args.model_path,
                "prompt": prompt_ids,
                "max_tokens": max_tokens,
                "temperature": 0,
            }, timeout=120)
            d = resp.json()
            if "error" in d:
                print(f"  {pt_file.name}: API error: {d[error]}")
                continue

            # Get generated token ids by re-tokenizing
            gen_text = d["choices"][0]["text"]
            gen_tokens = d["usage"]["completion_tokens"]

        except Exception as e:
            print(f"  {pt_file.name}: {e}")
            continue

        # Save intermediate result
        with open(str(out_path), "w") as f:
            json.dump({
                "file": pt_file.name,
                "prompt_ids": prompt_ids,
                "gen_text": gen_text,
                "gen_tokens": gen_tokens,
                "orig_resp_len": orig_resp_len,
            }, f)

    print(f"Generation done. Results in {output_dir}")


def step_extract(args):
    """Extract hidden states for greedy sequences."""
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer

    output_dir = Path(args.output_dir)

    # Load tokenizer to re-encode generated text
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Collect all greedy sequences
    json_files = sorted(output_dir.glob("data_*_greedy.json"))
    print(f"Found {len(json_files)} greedy generation results")

    # Build full token sequences (prompt + greedy response)
    sequences = []
    for jf in json_files:
        with open(str(jf)) as f:
            d = json.load(f)

        prompt_ids = d["prompt_ids"]
        gen_text = d["gen_text"]

        # Re-encode generated text to get token ids
        gen_ids = tok.encode(gen_text, add_special_tokens=False)

        full_ids = prompt_ids + gen_ids
        # Truncate to max_model_len
        full_ids = full_ids[:args.max_model_len - 1]

        # Create loss_mask: 0 for prompt, 1 for response
        loss_mask = [0] * len(prompt_ids) + [1] * len(gen_ids)
        loss_mask = loss_mask[:len(full_ids)]

        sequences.append({
            "file": d["file"],
            "full_ids": full_ids,
            "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
            "prompt_len": len(prompt_ids),
        })

    print(f"Prepared {len(sequences)} sequences for hidden state extraction")

    # Extract hidden states using VllmHiddenStatesGenerator
    from speculators.data_generation.vllm_hidden_states_generator import VllmHiddenStatesGenerator

    print(f"Loading INT4 model from {args.model_path}...")
    generator = VllmHiddenStatesGenerator(
        model_path=args.model_path,
        layer_ids=args.layer_ids,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.85,
    )
    print("Generator ready")

    # Load verifier weights for top-K logits
    import safetensors.torch
    import torch.nn as nn
    model_index_path = Path(args.model_path) / "model.safetensors.index.json"
    vw = {}
    with open(str(model_index_path)) as f2:
        idx = json.load(f2)
    needed = {"language_model.lm_head.weight", "language_model.model.norm.weight"}
    for shard in set(v for k,v in idx["weight_map"].items() if k in needed):
        w = safetensors.torch.load_file(str(Path(args.model_path) / shard))
        vw.update({k: w[k] for k in w if k in needed})
    hs = vw["language_model.model.norm.weight"].shape[0]
    vs = vw["language_model.lm_head.weight"].shape[0]
    # Get RMSNorm eps from model config (not hardcoded)
    from transformers import AutoConfig
    _cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    _text_cfg = getattr(_cfg, 'text_config', _cfg)
    _eps = getattr(_text_cfg, 'rms_norm_eps', 1e-6)
    v_norm = nn.modules.normalization.RMSNorm(hs, eps=_eps).cuda().to(torch.bfloat16)
    print(f"  RMSNorm eps={_eps}")
    v_norm.load_state_dict({"weight": vw["language_model.model.norm.weight"].to(torch.bfloat16)})
    v_lm_w = vw["language_model.lm_head.weight"].to(torch.bfloat16).cuda()
    TOP_K = 100
    print(f"Verifier logits: hidden={hs}, vocab={vs}, top_k={TOP_K}")

    # Process in batches
    processed = 0
    for batch_start in tqdm(range(0, len(sequences), args.batch_size), desc="Extracting"):
        batch = sequences[batch_start:batch_start + args.batch_size]
        token_ids = [s["full_ids"] for s in batch]

        try:
            results = generator.generate(token_ids)
        except Exception as e:
            print(f"Batch {batch_start} failed: {e}")
            continue

        # Reorder results (generator sorts by str(idx))
        sorted_idx = sorted(range(len(batch)), key=lambda i: str(i))
        reordered = [None] * len(batch)
        for k, orig in enumerate(sorted_idx):
            reordered[orig] = results[k]

        for seq, result in zip(batch, reordered):
            out_name = seq["file"].replace(".pt", ".pt")
            out_path = output_dir / out_name
            last_hs = result["hidden_states"][-1]
            with torch.no_grad():
                normed = v_norm(last_hs.unsqueeze(0).cuda().to(torch.bfloat16))
                lgt = torch.nn.functional.linear(normed[0], v_lm_w)
                tk_v, tk_i = torch.topk(lgt, TOP_K, dim=-1)
            torch.save({
                "input_ids": result["input_ids"],
                "hidden_states": result["hidden_states"],
                "loss_mask": seq["loss_mask"],
                "top_logits_values": tk_v.cpu().to(torch.bfloat16),
                "top_logits_indices": tk_i.cpu().to(torch.int32),
            }, str(out_path))
            processed += 1

    print(f"Done: processed={processed}")


def main():
    args = parse_args()
    if args.step in ("generate", "both"):
        step_generate(args)
    if args.step in ("extract", "both"):
        step_extract(args)


if __name__ == "__main__":
    main()
