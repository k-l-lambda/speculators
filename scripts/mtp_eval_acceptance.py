#!/usr/bin/env python3
"""
Phase 2: Standalone MTP forward pass + acceptance rate computation.

Supports two weight formats:
  1. K2.5-MTP: separate mtp.safetensors with INT4 GPTQ quantized experts
  2. DeepSeek-V3: MTP layer (layer 61) embedded in model shards with FP8 experts

Usage (K2.5):
    python mtp_eval_acceptance.py \
        --data-dir /data/mtp_eval/ \
        --mtp-weights /data/models/Kimi-K2.5-MTP/mtp.safetensors \
        --model-config /data/models/Kimi-K2.5-MTP/ \
        --output /data/mtp_eval/results.json

Usage (V3):
    python mtp_eval_acceptance.py \
        --data-dir /data/mtp_eval_v3/ \
        --model-dir /path/to/DeepSeek-V3/ \
        --output /data/mtp_eval_v3/results.json
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2: MTP acceptance rate evaluation"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Directory with Phase 1 .pt files",
    )
    # K2.5 mode: separate mtp.safetensors
    parser.add_argument(
        "--mtp-weights", type=str, default=None,
        help="Path to mtp.safetensors (K2.5 INT4 GPTQ mode)",
    )
    parser.add_argument(
        "--model-config", type=str, default=None,
        help="Path to directory with config.json and modeling_deepseek.py (K2.5 mode)",
    )
    # V3 mode: extract MTP from model directory
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Path to model directory with safetensors shards (V3 mode)",
    )
    parser.add_argument(
        "--mtp-layer-idx", type=int, default=61,
        help="MTP layer index in model (default: 61)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output JSON results file",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device to run on (default: cuda:0)",
    )
    return parser.parse_args()


# ============================================================
# Dequantization: INT4 GPTQ (K2.5)
# ============================================================

def dequant_int4_gptq(weight_packed, weight_scale, weight_shape, group_size=32):
    """Dequantize INT4 GPTQ packed weights to BF16."""
    out_f = weight_shape[0].item()
    in_f = weight_shape[1].item()
    PACK_FACTOR = 8

    unpacked = [(weight_packed >> (i * 4)) & 0xF for i in range(PACK_FACTOR)]
    w = torch.stack(unpacked, dim=-1).reshape(out_f, -1)[:, :in_f]
    w_signed = w.float() - 8.0
    w_grouped = w_signed.reshape(out_f, -1, group_size)
    scales = weight_scale.float().unsqueeze(-1)
    return (w_grouped * scales).reshape(out_f, -1)[:, :in_f].bfloat16()


# ============================================================
# Dequantization: FP8 e4m3 with block-wise scales (V3)
# ============================================================

def dequant_fp8_block(weight_fp8, weight_scale_inv, block_size=128):
    """Dequantize FP8 e4m3 block-quantized weights to BF16.

    Args:
        weight_fp8: FP8 tensor [out_features, in_features]
        weight_scale_inv: Inverse scales per block [ceil(out/block), ceil(in/block)]
        block_size: Block size for quantization (default: 128)
    Returns:
        BF16 tensor [out_features, in_features]
    """
    out_f, in_f = weight_fp8.shape
    w = weight_fp8.float()

    # Expand block scales to match weight dimensions
    scales = weight_scale_inv.float()
    scales_expanded = scales.repeat_interleave(block_size, dim=0)[:out_f, :]
    scales_expanded = scales_expanded.repeat_interleave(block_size, dim=1)[:, :in_f]

    return (w * scales_expanded).bfloat16()


# ============================================================
# Weight loading: K2.5 mode (separate mtp.safetensors)
# ============================================================

def load_mtp_state_dict_k25(mtp_weights_path, device="cpu"):
    """Load and dequantize K2.5 MTP weights from mtp.safetensors."""
    log.info(f"Loading K2.5 MTP weights from {mtp_weights_path}...")
    f = safe_open(mtp_weights_path, framework="pt", device=str(device))
    keys = list(f.keys())
    log.info(f"Total keys in mtp.safetensors: {len(keys)}")

    PREFIX = "model.layers.61."
    state_dict = {}
    expert_pattern = re.compile(
        r"mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight_packed|weight_scale|weight_shape)"
    )

    expert_parts = {}

    for key in tqdm(keys, desc="Loading weights"):
        short_key = key.removeprefix(PREFIX)
        tensor = f.get_tensor(key)

        m = expert_pattern.match(short_key)
        if m:
            expert_id, proj_name, part_name = m.group(1), m.group(2), m.group(3)
            ek = (int(expert_id), proj_name)
            if ek not in expert_parts:
                expert_parts[ek] = {}
            expert_parts[ek][part_name] = tensor
        else:
            state_dict[short_key] = tensor

    log.info(f"Dequantizing {len(expert_parts)} expert projections (INT4 GPTQ)...")
    for (expert_id, proj_name), parts in tqdm(
        sorted(expert_parts.items()), desc="Dequantizing experts"
    ):
        w = dequant_int4_gptq(
            parts["weight_packed"],
            parts["weight_scale"],
            parts["weight_shape"],
        )
        state_dict[f"mlp.experts.{expert_id}.{proj_name}.weight"] = w

    log.info(f"State dict has {len(state_dict)} entries")
    return state_dict


# ============================================================
# Weight loading: V3 mode (extract MTP layer from model shards)
# ============================================================

def load_mtp_state_dict_v3(model_dir, mtp_layer_idx=61, device="cpu"):
    """Load and dequantize V3 MTP weights from model shard files."""
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"

    log.info(f"Loading V3 MTP weights from {model_dir} (layer {mtp_layer_idx})...")

    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Find all keys for the MTP layer
    prefix = f"model.layers.{mtp_layer_idx}."
    mtp_keys = {k: v for k, v in weight_map.items() if k.startswith(prefix)}
    # Also grab lm_head for comparison
    if "lm_head.weight" in weight_map:
        mtp_keys["lm_head.weight"] = weight_map["lm_head.weight"]

    log.info(f"Found {len(mtp_keys)} MTP-related keys across shards")

    # Group by shard file
    shards = {}
    for k, v in mtp_keys.items():
        shards.setdefault(v, []).append(k)

    state_dict = {}
    expert_fp8_parts = {}  # (expert_id, proj_name) -> {weight, scale_inv}
    expert_pattern = re.compile(
        r"mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)"
    )
    # Non-expert FP8 patterns: collect weight + scale_inv pairs
    fp8_non_expert = {}  # short_key_base -> {weight, scale_inv}

    for shard_file, keys in sorted(shards.items()):
        shard_path = model_dir / shard_file
        log.info(f"Loading {shard_file} ({len(keys)} keys)...")
        f = safe_open(str(shard_path), framework="pt", device=str(device))

        for key in keys:
            tensor = f.get_tensor(key)

            if key == "lm_head.weight":
                state_dict["_lm_head.weight"] = tensor
                continue

            short_key = key.removeprefix(prefix)

            m = expert_pattern.match(short_key)
            if m:
                expert_id, proj_name, part = m.group(1), m.group(2), m.group(3)
                ek = (int(expert_id), proj_name)
                if ek not in expert_fp8_parts:
                    expert_fp8_parts[ek] = {}
                if part == "weight_scale_inv":
                    expert_fp8_parts[ek]["scale_inv"] = tensor
                else:
                    expert_fp8_parts[ek]["weight"] = tensor
            elif short_key.endswith(".weight_scale_inv"):
                base_key = short_key.removesuffix(".weight_scale_inv")
                fp8_non_expert.setdefault(base_key, {})["scale_inv"] = tensor
            elif short_key.endswith(".weight"):
                base_key = short_key.removesuffix(".weight")
                # Check if this might be an FP8 weight (corresponding scale_inv exists)
                scale_full_key = key + "_scale_inv"
                if scale_full_key in weight_map:
                    fp8_non_expert.setdefault(base_key, {})["weight"] = tensor
                else:
                    state_dict[short_key] = tensor
            else:
                state_dict[short_key] = tensor

    # Dequantize non-expert FP8 weights
    for base_key, parts in fp8_non_expert.items():
        if "weight" in parts and "scale_inv" in parts:
            w = dequant_fp8_block(parts["weight"], parts["scale_inv"])
            state_dict[f"{base_key}.weight"] = w
            log.info(f"  Dequantized FP8: {base_key}.weight {parts['weight'].shape}")
        elif "weight" in parts:
            state_dict[f"{base_key}.weight"] = parts["weight"]

    # Dequantize expert FP8 weights
    log.info(f"Dequantizing {len(expert_fp8_parts)} expert projections (FP8)...")
    for (expert_id, proj_name), parts in tqdm(
        sorted(expert_fp8_parts.items()), desc="Dequantizing experts"
    ):
        if "weight" in parts and "scale_inv" in parts:
            w = dequant_fp8_block(parts["weight"], parts["scale_inv"])
        elif "weight" in parts:
            w = parts["weight"].bfloat16()
        else:
            log.warning(f"Missing weight for expert {expert_id}.{proj_name}")
            continue
        state_dict[f"mlp.experts.{expert_id}.{proj_name}.weight"] = w

    log.info(f"State dict has {len(state_dict)} entries")
    return state_dict


# ============================================================
# Model building
# ============================================================

def build_mtp_model(config_dir, state_dict, device):
    """Build standalone MTP layer from DeepSeek config + dequantized weights."""
    config_dir = str(config_dir)
    if config_dir not in sys.path:
        sys.path.insert(0, config_dir)

    from configuration_deepseek import DeepseekV3Config
    from modeling_deepseek import DeepseekV3DecoderLayer, DeepseekV3RMSNorm

    with open(Path(config_dir) / "config.json") as f:
        full_config = json.load(f)
    text_config = full_config.get("text_config", full_config)

    config = DeepseekV3Config(**text_config)
    config._attn_implementation = "eager"
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size

    log.info(f"Building MTP layer: hidden_size={hidden_size}, vocab_size={vocab_size}")

    class MTPLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
            self.enorm = DeepseekV3RMSNorm(hidden_size, eps=config.rms_norm_eps)
            self.hnorm = DeepseekV3RMSNorm(hidden_size, eps=config.rms_norm_eps)
            self.eh_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            self.decoder_layer = DeepseekV3DecoderLayer(config, layer_idx=61)
            self.shared_head_norm = DeepseekV3RMSNorm(hidden_size, eps=config.rms_norm_eps)
            self.shared_head = nn.Linear(hidden_size, vocab_size, bias=False)

        def forward(self, h_t, x_next_ids, debug=False):
            batch_size, seq_len, _ = h_t.shape

            x_embed = self.enorm(self.embed_tokens(x_next_ids))
            h_norm = self.hnorm(h_t)
            hidden = self.eh_proj(torch.cat([x_embed, h_norm], dim=-1))

            if debug:
                log.info("  h_t: mean=%.4f std=%.4f" % (h_t.float().mean().item(), h_t.float().std().item()))
                log.info("  x_embed: mean=%.4f std=%.4f" % (x_embed.float().mean().item(), x_embed.float().std().item()))
                log.info("  eh_proj out: mean=%.4f std=%.4f" % (hidden.float().mean().item(), hidden.float().std().item()))

            position_ids = torch.arange(seq_len, device=h_t.device).unsqueeze(0).expand(batch_size, -1)

            causal_mask = torch.full(
                (batch_size, 1, seq_len, seq_len),
                torch.finfo(hidden.dtype).min,
                device=hidden.device, dtype=hidden.dtype
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)

            layer_out = self.decoder_layer(
                hidden_states=hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
            )
            hidden = layer_out[0]

            if debug:
                log.info("  decoder out: mean=%.4f std=%.4f" % (hidden.float().mean().item(), hidden.float().std().item()))

            logits = self.shared_head(self.shared_head_norm(hidden))
            return logits

    model = MTPLayer()

    key_mapping = {
        "embed_tokens.weight": "embed_tokens.weight",
        "enorm.weight": "enorm.weight",
        "hnorm.weight": "hnorm.weight",
        "eh_proj.weight": "eh_proj.weight",
        "shared_head.norm.weight": "shared_head_norm.weight",
        "shared_head.head.weight": "shared_head.weight",
        "input_layernorm.weight": "decoder_layer.input_layernorm.weight",
        "post_attention_layernorm.weight": "decoder_layer.post_attention_layernorm.weight",
    }

    new_state = {}
    for src_key, tensor in state_dict.items():
        if src_key.startswith("_"):  # Skip internal keys like _lm_head.weight
            continue
        if src_key in key_mapping:
            new_state[key_mapping[src_key]] = tensor
        elif src_key.startswith("self_attn."):
            new_state[f"decoder_layer.{src_key}"] = tensor
        elif src_key.startswith("mlp."):
            new_state[f"decoder_layer.{src_key}"] = tensor
        else:
            log.warning(f"Unmapped key: {src_key}")

    model_state = model.state_dict()
    missing = set(model_state.keys()) - set(new_state.keys())
    unexpected = set(new_state.keys()) - set(model_state.keys())
    if missing:
        log.warning(f"Missing keys ({len(missing)}): {sorted(missing)[:10]}...")
    if unexpected:
        log.warning(f"Unexpected keys ({len(unexpected)}): {sorted(unexpected)[:10]}...")

    model.load_state_dict(new_state, strict=False)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    log.info(f"MTP model loaded: {param_count / 1e9:.2f}B parameters")

    return model


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_acceptance(model, data_dir, device, output_path):
    """Run teacher-forced MTP evaluation and compute acceptance rates."""
    data_dir = Path(data_dir)
    pt_files = sorted(data_dir.glob("data_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    log.info(f"Found {len(pt_files)} data files")

    per_sample_results = []
    total_top1_correct = 0
    total_top5_correct = 0
    total_evaluated = 0

    for pt_file in tqdm(pt_files, desc="Evaluating acceptance"):
        sample = torch.load(str(pt_file), map_location="cpu", weights_only=True)
        input_ids = sample["input_ids"]
        hidden_states = sample["hidden_states"]
        loss_mask = sample["loss_mask"]

        seq_len = len(input_ids)
        if seq_len < 3:
            continue

        h_t = hidden_states[:-2].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        x_next = input_ids[1:-1].unsqueeze(0).to(device=device)
        targets = input_ids[2:].to(device=device)
        mask = loss_mask[2:].to(device=device)

        if mask.sum() == 0:
            continue

        is_debug = len(per_sample_results) < 1
        logits = model(h_t, x_next, debug=is_debug).squeeze(0).float()
        torch.cuda.empty_cache()

        preds = logits.argmax(dim=-1)
        top1_match = (preds == targets) & (mask == 1)
        top1_correct = top1_match.sum().item()

        if len(per_sample_results) < 3:
            log.info(
                "DEBUG sample %d: logits range=[%.2f, %.2f], std=%.4f, "
                "top1=%d/%d, pred[:5]=%s, target[:5]=%s",
                len(per_sample_results), logits.min().item(), logits.max().item(),
                logits.std().item(), top1_correct, mask.sum().item(),
                preds[:5].tolist(), targets[:5].tolist()
            )

        top5_preds = logits.topk(5, dim=-1).indices
        top5_match = (top5_preds == targets.unsqueeze(-1)).any(dim=-1) & (mask == 1)
        top5_correct = top5_match.sum().item()

        n_evaluated = mask.sum().item()
        total_top1_correct += top1_correct
        total_top5_correct += top5_correct
        total_evaluated += n_evaluated

        per_sample_results.append({
            "file": pt_file.name,
            "seq_len": seq_len,
            "n_response_tokens": int(n_evaluated),
            "top1_correct": int(top1_correct),
            "top5_correct": int(top5_correct),
            "top1_rate": round(top1_correct / n_evaluated, 4) if n_evaluated > 0 else 0,
            "top5_rate": round(top5_correct / n_evaluated, 4) if n_evaluated > 0 else 0,
        })

    overall_top1 = total_top1_correct / total_evaluated if total_evaluated > 0 else 0
    overall_top5 = total_top5_correct / total_evaluated if total_evaluated > 0 else 0

    results = {
        "overall_top1_acceptance": round(overall_top1, 4),
        "overall_top5_acceptance": round(overall_top5, 4),
        "num_samples": len(per_sample_results),
        "total_evaluated_tokens": total_evaluated,
        "total_top1_correct": total_top1_correct,
        "total_top5_correct": total_top5_correct,
        "per_sample": per_sample_results,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info("=" * 60)
    log.info("MTP Acceptance Rate Results")
    log.info("=" * 60)
    log.info(f"  Samples:     {len(per_sample_results)}")
    log.info(f"  Tokens:      {total_evaluated}")
    log.info(f"  Top-1:       {overall_top1:.4f} ({total_top1_correct}/{total_evaluated})")
    log.info(f"  Top-5:       {overall_top5:.4f} ({total_top5_correct}/{total_evaluated})")
    log.info(f"  Results:     {output_path}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    # Determine mode
    if args.model_dir:
        mode = "v3"
        log.info("Mode: DeepSeek-V3 (FP8, model shards)")
    elif args.mtp_weights:
        mode = "k25"
        log.info("Mode: Kimi-K2.5 (INT4 GPTQ, separate mtp.safetensors)")
    else:
        print("Error: must specify either --model-dir (V3) or --mtp-weights (K2.5)")
        sys.exit(1)

    # Default model-config
    if args.model_config is None:
        if mode == "v3":
            args.model_config = args.model_dir
        else:
            args.model_config = str(Path(__file__).resolve().parent / "k2_mtp_config")

    log.info("=" * 60)
    log.info("Phase 2: MTP Acceptance Rate Evaluation")
    log.info("=" * 60)

    # Load weights
    if mode == "v3":
        state_dict = load_mtp_state_dict_v3(
            args.model_dir, mtp_layer_idx=args.mtp_layer_idx, device="cpu"
        )
    else:
        state_dict = load_mtp_state_dict_k25(args.mtp_weights, device="cpu")

    # Build model
    model = build_mtp_model(args.model_config, state_dict, args.device)
    del state_dict

    # Evaluate
    results = evaluate_acceptance(model, args.data_dir, args.device, args.output)

    log.info("Phase 2 complete!")
    return results


if __name__ == "__main__":
    main()
