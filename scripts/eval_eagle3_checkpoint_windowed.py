#!/usr/bin/env python3
"""Evaluate Eagle3 checkpoints on full samples and token windows.

This keeps the same sample set for every length range by truncating long
sequences to a prefix and restricting metrics to a target token window.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors
from tqdm import tqdm

os.environ["TRUST_REMOTE_CODE"] = "1"

from speculators.models.eagle3 import Eagle3SpeculatorConfig
from speculators.models.eagle3.core import Eagle3DraftModel
from speculators.train.data import shift_batch, standardize_data_v1

WINDOW_ENDS = [1024, 2048, 4096, 8192]


def load_eagle3_from_checkpoint(checkpoint_dir: str, d2t=None, t2d=None) -> Eagle3DraftModel:
    checkpoint_path = Path(checkpoint_dir)
    config = Eagle3SpeculatorConfig.from_pretrained(str(checkpoint_path))
    model = Eagle3DraftModel(config, t2d=t2d, d2t=d2t)

    shard_files = sorted(checkpoint_path.glob("model-*.safetensors"))
    if not shard_files:
        shard_files = [checkpoint_path / "model.safetensors"]

    print(f"  Loading {len(shard_files)} safetensors shard(s)...")
    state_dict = {}
    for shard in shard_files:
        state_dict.update(load_safetensors(str(shard), device="cpu"))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)} (expected for verifier weights)")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}...")
    return model


def prepare_data(raw: dict, apply_shift: bool) -> dict | None:
    try:
        data = standardize_data_v1(raw)
    except Exception:
        return None

    if apply_shift:
        data["position_ids"] = torch.arange(len(data["input_ids"]))
        data["lengths"] = torch.tensor([len(data["input_ids"])])
        data = shift_batch(data)
    return data


def build_window_sample(data: dict, start: int, end: int) -> dict | None:
    seq_len = len(data["input_ids"])
    end = min(end, seq_len)
    if end <= 1 or start >= end:
        return None

    sample = {}
    for key, value in data.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key == "lengths":
            sample[key] = torch.tensor([end], dtype=value.dtype)
        elif key == "position_ids":
            sample[key] = value[:end]
        elif value.ndim >= 1 and value.shape[0] == seq_len:
            sample[key] = value[:end].clone()
        else:
            sample[key] = value.clone()

    if "loss_mask" not in sample or int(sample["loss_mask"].sum()) == 0:
        return None

    sample["loss_mask"][:start] = 0
    if int(sample["loss_mask"].sum()) == 0:
        return None
    return sample


def eval_one(model, device, data: dict, ttt_steps: int):
    batch = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0).to(
                device=device,
                dtype=torch.bfloat16 if value.is_floating_point() else value.dtype,
            )

    if "lengths" not in batch:
        batch["lengths"] = torch.tensor([len(data["input_ids"])], device=device)
    if "position_ids" not in batch:
        batch["position_ids"] = torch.arange(len(data["input_ids"]), device=device).unsqueeze(0)

    _, _, metrics = model(**batch, ttt_steps=ttt_steps)
    result = {}
    for key, value in metrics.items():
        result[key] = value.item() if isinstance(value, torch.Tensor) else value
    return result


def accumulate(target: dict, metrics: dict):
    for key, value in metrics.items():
        target[key] = target.get(key, 0.0) + value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--d2t-path", default=None)
    parser.add_argument("--t2d-path", default=None)
    parser.add_argument("--ttt-steps", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--apply-shift", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    d2t = t2d = None
    if args.d2t_path and args.t2d_path:
        d2t = torch.from_numpy(np.load(args.d2t_path))
        t2d = torch.from_numpy(np.load(args.t2d_path))

    print(f"Loading Eagle3 from {args.checkpoint}...")
    model = load_eagle3_from_checkpoint(args.checkpoint, d2t=d2t, t2d=t2d)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    if model.d2t is not None:
        model.d2t = model.d2t.to(device)
    if model.t2d is not None:
        model.t2d = model.t2d.to(device)

    pt_files = sorted(Path(args.data_dir).glob("data_*.pt"), key=lambda pp: int(pp.stem.split("_")[1]))
    print(f"Found {len(pt_files)} data files in {args.data_dir}")

    all_metrics = {}
    full_count = 0
    window_metrics = {
        (0, 1024): {},
        (1024, 2048): {},
        (2048, 4096): {},
        (4096, 8192): {},
    }
    window_counts = {key: 0 for key in window_metrics}
    prefix_metrics = {end: {} for end in WINDOW_ENDS}
    prefix_counts = {end: 0 for end in WINDOW_ENDS}
    errors = []

    with torch.no_grad():
        for idx, pt_file in enumerate(tqdm(pt_files, desc="Evaluating"), start=1):
            raw = torch.load(str(pt_file), map_location="cpu", weights_only=False)
            data = prepare_data(raw, args.apply_shift)
            if data is None:
                continue

            try:
                metrics = eval_one(model, device, data, args.ttt_steps)
            except Exception as exc:
                errors.append((pt_file.name, str(exc)))
                print(f"  Error full {pt_file.name}: {exc}")
                continue

            accumulate(all_metrics, metrics)
            full_count += 1

            seq_len = len(data["input_ids"])
            for end in WINDOW_ENDS:
                prefix = build_window_sample(data, 0, end)
                if prefix is None:
                    continue
                try:
                    prefix_result = eval_one(model, device, prefix, args.ttt_steps)
                except Exception as exc:
                    errors.append((f"{pt_file.name}<={end}", str(exc)))
                    print(f"  Error prefix {pt_file.name} <= {end}: {exc}")
                    continue
                accumulate(prefix_metrics[end], prefix_result)
                prefix_counts[end] += 1

            for start, end in window_metrics:
                if seq_len <= start:
                    continue
                sample = build_window_sample(data, start, end)
                if sample is None:
                    continue
                try:
                    region_result = eval_one(model, device, sample, args.ttt_steps)
                except Exception as exc:
                    errors.append((f"{pt_file.name}[{start}:{end}]", str(exc)))
                    print(f"  Error window {pt_file.name} [{start}, {end}): {exc}")
                    continue
                accumulate(window_metrics[(start, end)], region_result)
                window_counts[(start, end)] += 1

            if idx % 100 == 0:
                torch.cuda.empty_cache()

    if full_count == 0:
        print("No valid samples evaluated!")
        return

    avg = {key: value / full_count for key, value in all_metrics.items()}
    acc_keys = sorted(key for key in avg if key.startswith("cond_acc"))

    print("\n" + "=" * 60)
    print("Eagle3 Evaluation Results")
    print("=" * 60)
    print(f"  Checkpoint:    {args.checkpoint}")
    print(f"  Data:          {args.data_dir} ({full_count}/{len(pt_files)} full samples)")
    print(f"  ttt_steps:     {args.ttt_steps}")
    for key in sorted(avg):
        print(f"  {key:30s}: {avg[key]:.4f}")

    print("\n" + "-" * 60)
    print("Cumulative prefix metrics (truncate long samples, keep same sample pool)")
    print("-" * 60)
    header = f"  {'Prefix':>10s}  {'N':>5s}"
    for key in acc_keys:
        header += f"  {key:>10s}"
    print(header)
    for end in WINDOW_ENDS:
        n = prefix_counts[end]
        if n == 0:
            continue
        row = f"  {'<=' + str(end):>10s}  {n:>5d}"
        for key in acc_keys:
            row += f"  {prefix_metrics[end].get(key, 0.0) / n:>10.4f}"
        print(row)

    print("\n" + "-" * 60)
    print("Region metrics (same sample, same context, different token windows)")
    print("-" * 60)
    header = f"  {'Window':>12s}  {'N':>5s}"
    for key in acc_keys:
        header += f"  {key:>10s}"
    print(header)
    for start, end in window_metrics:
        n = window_counts[(start, end)]
        if n == 0:
            continue
        row = f"  {f'[{start},{end})':>12s}  {n:>5d}"
        for key in acc_keys:
            row += f"  {window_metrics[(start, end)].get(key, 0.0) / n:>10.4f}"
        print(row)

    if errors:
        print("\n" + "-" * 60)
        print(f"Errors: {len(errors)}")
        for name, message in errors[:20]:
            print(f"  {name}: {message}")


if __name__ == "__main__":
    main()
