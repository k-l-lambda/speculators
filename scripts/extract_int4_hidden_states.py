#!/usr/bin/env python3
"""Extract INT4 hidden states using VllmHiddenStatesGenerator as __main__.

Key: Do NOT import torch before vLLM to avoid CUDA fork issue.
Run as: python3 extract_int4_vllm_main.py (standalone, not imported)
"""
import os
import sys
import argparse

# CRITICAL: Do NOT import torch here. Let vLLM initialize CUDA first.

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/data/models/Kimi-K2.5-MTP")
    parser.add_argument("--data-dir", type=str, default="/data/datasets/apilog_k25_eagle3/val_5k")
    parser.add_argument("--output-dir", type=str, default="/tmp/val_5k_int4")
    parser.add_argument("--layer-ids", type=int, nargs="+", default=[2, 30, 58, 60])
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # Import torch AFTER parsing args (before vLLM is fine if we don't touch CUDA)
    import torch
    from pathlib import Path
    from tqdm import tqdm

    # Now import vLLM - it will initialize CUDA via multiprocessing spawn
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from speculators.data_generation.vllm_hidden_states_generator import VllmHiddenStatesGenerator

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(data_dir.glob("data_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if args.max_samples:
        pt_files = pt_files[:args.max_samples]
    print(f"Found {len(pt_files)} .pt files", flush=True)

    print(f"Loading model from {args.model_path} TP={args.tensor_parallel_size}...", flush=True)
    generator = VllmHiddenStatesGenerator(
        model_path=args.model_path,
        layer_ids=args.layer_ids,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.85,
    )
    print("Generator ready", flush=True)

    processed = 0
    skipped = 0
    for batch_start in tqdm(range(0, len(pt_files), args.batch_size), desc="Batches"):
        batch_files = pt_files[batch_start:batch_start + args.batch_size]

        batch_data = []
        for pt_file in batch_files:
            sample = torch.load(str(pt_file), map_location="cpu", weights_only=True)
            input_ids = sample["input_ids"]
            if len(input_ids) < 3:
                skipped += 1
                continue
            input_ids = input_ids[:args.max_model_len - 1]
            batch_data.append({
                "file": pt_file,
                "input_ids": input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids,
                "loss_mask": sample.get("loss_mask"),
            })

        if not batch_data:
            continue

        token_ids = [d["input_ids"] for d in batch_data]
        try:
            results = generator.generate(token_ids)
        except Exception as e:
            print(f"Batch {batch_start} failed: {e}", flush=True)
            continue

        for data, result in zip(batch_data, results):
            out_path = output_dir / data["file"].name
            torch.save({
                "input_ids": result["input_ids"],
                "hidden_states": result["hidden_states"],
                "loss_mask": result.get("loss_mask", data["loss_mask"]),
            }, str(out_path))
            processed += 1

    print(f"Done: processed={processed}, skipped={skipped}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    # generator.shutdown()  # Not implemented


if __name__ == "__main__":
    main()
