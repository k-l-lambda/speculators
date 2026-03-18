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
        --output /tmp/results.json

Usage (V3):
    python mtp_eval_acceptance.py \
        --data-dir /data/mtp_eval_v3/ \
        --model-dir /data/.cache_claude/huggingface/hub/models--deepseek-ai--DeepSeek-V3/snapshots/.../ \
        --output /tmp/results_v3.json
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
    # K2.5 mode
    parser.add_argument(
        "--mtp-weights", type=str, default=None,
        help="Path to mtp.safetensors (K2.5 INT4 GPTQ mode)",
    )
    parser.add_argument(
        "--model-config", type=str, default=None,
        help="Path to directory with config.json and modeling_deepseek.py "
             "(K2.5: defaults to scripts/k2_mtp_config/; V3: defaults to --model-dir)",
    )
    # Speculators trained checkpoint mode
    parser.add_argument(
        "--speculators-checkpoint", type=str, default=None,
        help="Path to speculators checkpoint directory (BF16 shards, no dequant)",
    )
    parser.add_argument(
        "--verifier-name-or-path", type=str, default=None,
        help="Path to verifier model (K2.5) for loading frozen decoder weights in speculators mode",
    )
    # V3 mode
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Path to model directory with safetensors shards (V3 FP8 mode)",
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
    parser.add_argument(
        "--zero-position0-embed", action="store_true",
        help="Zero position-0 token embeddings (simulate vLLM behavior)",
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
# Dequantization: FP8 e4m3 block-wise scales (V3)
# ============================================================

def dequant_fp8_block(weight_fp8, weight_scale_inv, block_size=128):
    """Dequantize FP8 e4m3 block-quantized weights to BF16.

    Args:
        weight_fp8: FP8 tensor [out_features, in_features]
        weight_scale_inv: Inverse scales [ceil(out/block), ceil(in/block)]
        block_size: Block size (default: 128)
    """
    out_f, in_f = weight_fp8.shape
    w = weight_fp8.float()
    scales = weight_scale_inv.float()

    # Expand block scales to full weight shape
    scales_expanded = scales.repeat_interleave(block_size, dim=0)[:out_f, :]
    scales_expanded = scales_expanded.repeat_interleave(block_size, dim=1)[:, :in_f]

    return (w * scales_expanded).bfloat16()


# ============================================================
# Weight loading: K2.5 mode (separate mtp.safetensors, INT4 GPTQ)
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
            parts["weight_packed"], parts["weight_scale"], parts["weight_shape"]
        )
        state_dict[f"mlp.experts.{expert_id}.{proj_name}.weight"] = w

    log.info(f"State dict has {len(state_dict)} entries")
    return state_dict


# ============================================================
# Weight loading: V3 mode (extract MTP layer from model shards, FP8)
# ============================================================

def load_mtp_state_dict_v3(model_dir, mtp_layer_idx=61, device="cpu"):
    """Load and dequantize V3 MTP weights from model shard files."""
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"

    log.info(f"Loading V3 MTP weights from {model_dir} (layer {mtp_layer_idx})...")

    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    prefix = f"model.layers.{mtp_layer_idx}."
    mtp_keys = {k: v for k, v in weight_map.items() if k.startswith(prefix)}
    log.info(f"Found {len(mtp_keys)} MTP-related keys")

    # Group by shard file
    shards = {}
    for k, v in mtp_keys.items():
        shards.setdefault(v, []).append(k)

    state_dict = {}
    expert_fp8 = {}   # (expert_id, proj_name) -> {weight, scale_inv}
    fp8_weights = {}  # base_key -> {weight, scale_inv}

    expert_pattern = re.compile(
        r"mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight_scale_inv|weight)"
    )

    for shard_file, keys in sorted(shards.items()):
        shard_path = model_dir / shard_file
        log.info(f"  Loading {shard_file} ({len(keys)} keys)...")
        sf = safe_open(str(shard_path), framework="pt", device=str(device))

        for key in keys:
            tensor = sf.get_tensor(key)
            short_key = key.removeprefix(prefix)

            m = expert_pattern.match(short_key)
            if m:
                expert_id, proj_name, part = m.group(1), m.group(2), m.group(3)
                ek = (int(expert_id), proj_name)
                if ek not in expert_fp8:
                    expert_fp8[ek] = {}
                expert_fp8[ek][part] = tensor
            elif short_key.endswith(".weight_scale_inv"):
                base = short_key[: -len(".weight_scale_inv")]
                fp8_weights.setdefault(base, {})["scale_inv"] = tensor
            elif short_key.endswith(".weight"):
                base = short_key[: -len(".weight")]
                scale_key = key + "_scale_inv"
                if scale_key in weight_map:
                    fp8_weights.setdefault(base, {})["weight"] = tensor
                else:
                    # Plain BF16 weight (no FP8 quantization)
                    state_dict[short_key] = tensor
            else:
                state_dict[short_key] = tensor

    # Dequantize non-expert FP8 weights (attention projections etc.)
    for base, parts in fp8_weights.items():
        if "weight" in parts and "scale_inv" in parts:
            state_dict[f"{base}.weight"] = dequant_fp8_block(
                parts["weight"], parts["scale_inv"]
            )
        elif "weight" in parts:
            state_dict[f"{base}.weight"] = parts["weight"].bfloat16()

    # Dequantize expert FP8 weights
    log.info(f"Dequantizing {len(expert_fp8)} expert projections (FP8)...")
    for (expert_id, proj_name), parts in tqdm(
        sorted(expert_fp8.items()), desc="Dequantizing experts"
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
# Model building (shared for both K2.5 and V3)
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

    log.info(f"Building MTP layer: hidden_size={hidden_size}, vocab_size={vocab_size}, "
             f"n_routed_experts={getattr(config, 'n_routed_experts', 'N/A')}")

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
            if getattr(self, "zero_pos0", False):
                x_embed[:, 0, :] = 0.0
            h_norm = self.hnorm(h_t)
            hidden = self.eh_proj(torch.cat([x_embed, h_norm], dim=-1))

            if debug:
                log.info("  h_t: mean=%.4f std=%.4f" % (h_t.float().mean(), h_t.float().std()))
                log.info("  x_embed: mean=%.4f std=%.4f" % (x_embed.float().mean(), x_embed.float().std()))
                log.info("  eh_proj: mean=%.4f std=%.4f" % (hidden.float().mean(), hidden.float().std()))

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
                log.info("  decoder: mean=%.4f std=%.4f" % (hidden.float().mean(), hidden.float().std()))

            return self.shared_head(self.shared_head_norm(hidden))

    model = MTPLayer()

    # Map state dict keys to model parameter names
    key_mapping = {
        "embed_tokens.weight":             "embed_tokens.weight",
        "enorm.weight":                    "enorm.weight",
        "hnorm.weight":                    "hnorm.weight",
        "eh_proj.weight":                  "eh_proj.weight",
        "shared_head.norm.weight":         "shared_head_norm.weight",
        "shared_head.head.weight":         "shared_head.weight",
        "input_layernorm.weight":          "decoder_layer.input_layernorm.weight",
        "post_attention_layernorm.weight": "decoder_layer.post_attention_layernorm.weight",
    }

    new_state = {}
    for src_key, tensor in state_dict.items():
        if src_key in key_mapping:
            new_state[key_mapping[src_key]] = tensor
        elif src_key.startswith("self_attn."):
            new_state[f"decoder_layer.{src_key}"] = tensor
        elif src_key.startswith("mlp."):
            new_state[f"decoder_layer.{src_key}"] = tensor
        else:
            log.debug(f"Unmapped key: {src_key}")

    model_keys = set(model.state_dict().keys())
    missing = model_keys - set(new_state.keys())
    unexpected = set(new_state.keys()) - model_keys
    if missing:
        log.warning(f"Missing keys ({len(missing)}): {sorted(missing)[:5]}...")
    if unexpected:
        log.warning(f"Unexpected keys ({len(unexpected)}): {sorted(unexpected)[:5]}...")

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

    per_sample = []
    total_top1 = total_top5 = total_n = 0

    for pt_file in tqdm(pt_files, desc="Evaluating acceptance"):
        sample = torch.load(str(pt_file), map_location="cpu", weights_only=True)
        input_ids     = sample["input_ids"]
        hidden_states = sample["hidden_states"]
        # Handle 4-layer Eagle3 format: extract last layer for MTP
        # (standardize_data_mtp uses h[-1] = last/deepest layer, e.g. layer 60 for K2.5)
        if isinstance(hidden_states, list):
            hidden_states = hidden_states[-1]
        loss_mask     = sample["loss_mask"]
        # Align loss_mask length to input_ids (INT4 hidden states may be truncated)
        if loss_mask is not None and len(loss_mask) != len(input_ids):
            loss_mask = loss_mask[:len(input_ids)]

        seq_len = len(input_ids)
        if seq_len < 3:
            continue

        # Apply shift_batch alignment (same as training collate_fn)
        from speculators.train.data import shift_batch
        shifted = shift_batch({
            "input_ids": input_ids,
            "hidden_states": hidden_states,
            "verifier_last_hidden_states": hidden_states,
            "loss_mask": loss_mask,
            "lengths": torch.tensor([seq_len]),
            "position_ids": torch.arange(seq_len),
        })
        h_t = shifted["hidden_states"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        x_ids = shifted["input_ids"].unsqueeze(0).to(device=device)
        s_mask = shifted["loss_mask"].to(device=device)

        if s_mask.sum() == 0:
            continue

        # Compute targets with shift-aligned data (same as training)
        targets = torch.cat([shifted["input_ids"][1:], torch.zeros(1, dtype=shifted["input_ids"].dtype)]).to(device=device)
        adjusted_mask = s_mask.clone()
        adjusted_mask[-1] = 0
        mask = adjusted_mask.to(device=device)

        is_debug = len(per_sample) < 1
        logits = model(h_t, x_ids, debug=is_debug).squeeze(0).float()
        torch.cuda.empty_cache()

        preds = logits.argmax(dim=-1)
        top1_correct = ((preds == targets) & (mask == 1)).sum().item()

        if len(per_sample) < 3:
            log.info(
                "DEBUG sample %d: logits=[%.2f, %.2f] std=%.4f top1=%d/%d "
                "pred[:5]=%s target[:5]=%s",
                len(per_sample), logits.min(), logits.max(), logits.std(),
                top1_correct, mask.sum().item(),
                preds[:5].tolist(), targets[:5].tolist()
            )

        top5_preds = logits.topk(5, dim=-1).indices
        top5_correct = ((top5_preds == targets.unsqueeze(-1)).any(-1) & (mask == 1)).sum().item()
        n = mask.sum().item()

        total_top1 += top1_correct
        total_top5 += top5_correct
        total_n    += n

        per_sample.append({
            "file": pt_file.name,
            "seq_len": seq_len,
            "n_response_tokens": int(n),
            "top1_correct": int(top1_correct),
            "top5_correct": int(top5_correct),
            "top1_rate": round(top1_correct / n, 4) if n > 0 else 0,
            "top5_rate": round(top5_correct / n, 4) if n > 0 else 0,
        })

    overall_top1 = total_top1 / total_n if total_n > 0 else 0
    overall_top5 = total_top5 / total_n if total_n > 0 else 0

    results = {
        "overall_top1_acceptance": round(overall_top1, 4),
        "overall_top5_acceptance": round(overall_top5, 4),
        "num_samples":             len(per_sample),
        "total_evaluated_tokens":  total_n,
        "total_top1_correct":      total_top1,
        "total_top5_correct":      total_top5,
        "per_sample":              per_sample,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info("=" * 60)
    log.info("MTP Acceptance Rate Results")
    log.info("=" * 60)
    log.info(f"  Samples: {len(per_sample)}")
    log.info(f"  Tokens:  {total_n}")
    log.info(f"  Top-1:   {overall_top1:.4f} ({total_top1}/{total_n})")
    log.info(f"  Top-5:   {overall_top5:.4f} ({total_top5}/{total_n})")
    log.info(f"  Output:  {output_path}")
    return results



# ============================================================
# Load: Speculators checkpoint (BF16, no dequantization)
# ============================================================

def load_mtp_state_dict_speculators(checkpoint_dir, device="cpu"):
    import json
    from safetensors import safe_open
    from pathlib import Path
    ckpt = Path(checkpoint_dir)
    with open(ckpt / "model.safetensors.index.json") as f:
        index = json.load(f)
    raw = {}
    for shard in sorted(set(index["weight_map"].values())):
        log.info("  Loading shard %s...", shard)
        with safe_open(str(ckpt / shard), framework="pt", device=device) as f:
            for k in f.keys():
                raw[k] = f.get_tensor(k)
    # Remap: speculators keys -> MTPLayer keys
    remapped = {}
    for k, v in raw.items():
        if k == "shared_head.weight":
            remapped["shared_head.head.weight"] = v
        elif k == "shared_head_norm.weight":
            remapped["shared_head.norm.weight"] = v
        elif k.startswith("layers.0."):
            remapped["decoder_layer." + k[len("layers.0."):]] = v
        else:
            remapped[k] = v
    log.info("Loaded %d tensors from speculators checkpoint", len(remapped))
    return remapped


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if args.speculators_checkpoint:
        mode = "speculators"
    elif args.model_dir:
        mode = "v3"
    elif args.mtp_weights:
        mode = "k25"
    else:
        sys.exit("Error: specify --speculators-checkpoint, --model-dir, or --mtp-weights")

    # Default model-config directory
    if args.model_config is None:
        if mode == "v3":
            args.model_config = args.model_dir
        else:
            args.model_config = str(Path(__file__).resolve().parent / "k2_mtp_config")

    log.info("=" * 60)
    log.info(f"Phase 2: MTP Acceptance Rate Evaluation [{mode.upper()} mode]")
    log.info("=" * 60)

    if mode == "speculators":
        # Use MTPDraftModel which properly loads K2.5 frozen decoder weights
        from speculators.models.mtp import MTPDraftModel
        from transformers import AutoConfig
        from safetensors.torch import load_file as load_safetensors
        from pathlib import Path as _P

        verifier_path = args.verifier_name_or_path
        if verifier_path is None:
            verifier_path = "/data/.cache_claude/huggingface/hub/models--moonshotai--Kimi-K2.5/snapshots/54383e83fa343a1331754112fb9e3410c55efa2f"
            log.info("Using default verifier: %s", verifier_path)

        verifier_config = AutoConfig.from_pretrained(verifier_path, trust_remote_code=True)
        if hasattr(verifier_config, "text_config"):
            verifier_config = verifier_config.text_config

        model = MTPDraftModel.from_training_args(
            verifier_config=verifier_config,
            verifier_name_or_path=verifier_path,
            freeze_decoder=True,
            verifier_layer_idx=-1,
        )
        # Load speculators checkpoint (trainable weights only)
        ckpt_path = _P(args.speculators_checkpoint)
        sd = {}
        for f in sorted(ckpt_path.glob("model-*.safetensors")):
            sd.update(load_safetensors(str(f), device="cpu"))
        if not sd:
            sd = load_safetensors(str(ckpt_path / "model.safetensors"), device="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log.info("Loaded checkpoint: %d keys, %d missing (frozen decoder)", len(sd), len(missing))

        model = model.to(device=args.device, dtype=torch.bfloat16).eval()
        param_count = sum(p.numel() for p in model.parameters())
        log.info("MTP model loaded: %.2fB parameters", param_count / 1e9)
        state_dict = None
    elif mode == "v3":
        state_dict = load_mtp_state_dict_v3(
            args.model_dir, mtp_layer_idx=args.mtp_layer_idx, device="cpu"
        )
    else:
        state_dict = load_mtp_state_dict_k25(args.mtp_weights, device="cpu")

    if state_dict is not None:
        model = build_mtp_model(args.model_config, state_dict, args.device)
        del state_dict
        if args.zero_position0_embed:
            model.zero_pos0 = True
            log.info("Enabled position-0 embedding zeroing")

    results = evaluate_acceptance(model, args.data_dir, args.device, args.output)
    log.info("Phase 2 complete!")
    return results


if __name__ == "__main__":
    main()
