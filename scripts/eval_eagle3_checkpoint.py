#!/usr/bin/env python3
"""Quick Eagle3 checkpoint evaluation on a val data directory.

Loads model using safetensors state dict (same path as trainer).
"""
import argparse, os, json, torch, numpy as np
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import load_file as load_safetensors

os.environ["TRUST_REMOTE_CODE"] = "1"

from speculators.models.eagle3.core import Eagle3DraftModel
from speculators.models.eagle3 import Eagle3SpeculatorConfig
from speculators.train.data import standardize_data_v1


def load_eagle3_from_checkpoint(checkpoint_dir: str, d2t=None, t2d=None,
                                  device="cpu") -> Eagle3DraftModel:
    """Load Eagle3 model from speculators checkpoint directory.

    Instantiates model on CPU then loads state dict (avoids meta-device issues).
    """
    checkpoint_path = Path(checkpoint_dir)

    # Load config
    config = Eagle3SpeculatorConfig.from_pretrained(str(checkpoint_path))

    # Create model on CPU (avoids meta-device issues)
    model = Eagle3DraftModel(config, t2d=t2d, d2t=d2t)

    # Find safetensors file(s)
    shard_files = sorted(checkpoint_path.glob("model-*.safetensors"))
    if not shard_files:
        # Single file
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--d2t-path", default=None)
    p.add_argument("--t2d-path", default=None)
    p.add_argument("--ttt-steps", type=int, default=3)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    device = torch.device(args.device)

    d2t = t2d = None
    if args.d2t_path and args.t2d_path:
        d2t = torch.from_numpy(np.load(args.d2t_path))
        t2d = torch.from_numpy(np.load(args.t2d_path))

    print(f"Loading Eagle3 from {args.checkpoint}...")
    model = load_eagle3_from_checkpoint(args.checkpoint, d2t=d2t, t2d=t2d)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    # Move d2t/t2d to device after model load
    if model.d2t is not None:
        model.d2t = model.d2t.to(device)
    if model.t2d is not None:
        model.t2d = model.t2d.to(device)

    print(f"Model on device. Params: {sum(p.numel() for p in model.parameters()):,}")

    pt_files = sorted(Path(args.data_dir).glob("data_*.pt"),
                      key=lambda pp: int(pp.stem.split("_")[1]))
    print(f"Found {len(pt_files)} data files in {args.data_dir}")

    all_metrics = {}
    num_valid = 0

    with torch.no_grad():
        for pt_file in tqdm(pt_files, desc="Evaluating"):
            raw = torch.load(str(pt_file), map_location="cpu", weights_only=False)
            try:
                data = standardize_data_v1(raw)
            except Exception as e:
                continue

            seq_len = len(data["input_ids"])
            if seq_len < 3 or data["loss_mask"].sum() == 0:
                continue

            batch = {}
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.unsqueeze(0).to(
                        device=device,
                        dtype=torch.bfloat16 if v.is_floating_point() else v.dtype
                    )

            batch["lengths"] = torch.tensor([seq_len], device=device)
            # Pass 0-indexed position_ids to avoid OOB in rotary (core.py uses 1+arange)
            batch["position_ids"] = torch.arange(seq_len, device=device).unsqueeze(0)

            try:
                _tokens, _loss, metrics = model(**batch, ttt_steps=args.ttt_steps)
            except Exception as e:
                print(f"  Error {pt_file.name}: {e}")
                continue

            for k, v in metrics.items():
                all_metrics[k] = all_metrics.get(k, 0.0) + (
                    v.item() if isinstance(v, torch.Tensor) else v
                )
            num_valid += 1

            if num_valid % 200 == 0:
                torch.cuda.empty_cache()

    if num_valid == 0:
        print("No valid samples evaluated!")
        return

    avg = {k: v / num_valid for k, v in all_metrics.items()}

    print("\n" + "=" * 60)
    print("Eagle3 Evaluation Results")
    print("=" * 60)
    print(f"  Checkpoint:    {args.checkpoint}")
    print(f"  Data:          {args.data_dir} ({num_valid}/{len(pt_files)} evaluated)")
    print(f"  ttt_steps:     {args.ttt_steps}")
    for k in sorted(avg.keys()):
        print(f"  {k:30s}: {avg[k]:.4f}")


if __name__ == "__main__":
    main()
