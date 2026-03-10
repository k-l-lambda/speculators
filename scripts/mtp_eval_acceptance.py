#!/usr/bin/env python3
"""
Phase 2: Standalone MTP forward pass + acceptance rate computation.

Loads INT4-quantized MTP weights from mtp.safetensors, dequantizes expert
weights to BF16, builds a standalone MTP layer, and runs teacher-forced
evaluation against Phase 1 hidden states.

Usage:
    python mtp_eval_acceptance.py \
        --data-dir /data/mtp_eval/ \
        --mtp-weights /data/models/Kimi-K2.5-MTP/mtp.safetensors \
        --model-config /data/models/Kimi-K2.5-MTP/ \
        --output /data/mtp_eval/results.json
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
    parser.add_argument(
        "--mtp-weights", type=str, required=True,
        help="Path to mtp.safetensors (INT4 GPTQ quantized)",
    )
    parser.add_argument(
        "--model-config", type=str, required=True,
        help="Path to directory with K2.5-MTP config.json and modeling_deepseek.py",
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


def dequant_int4_gptq(weight_packed, weight_scale, weight_shape, group_size=32):
    """Dequantize INT4 GPTQ packed weights to BF16.

    Args:
        weight_packed: Packed INT4 tensor [out_features, in_features // 8]
        weight_scale: Per-group scales [out_features, num_groups]
        weight_shape: Tensor with [out_features, in_features]
        group_size: Quantization group size
    """
    out_f = weight_shape[0].item()
    in_f = weight_shape[1].item()
    PACK_FACTOR = 8  # 8 INT4 values packed per INT32

    # Unpack: extract 8 4-bit values from each int32
    unpacked = [(weight_packed >> (i * 4)) & 0xF for i in range(PACK_FACTOR)]
    w = torch.stack(unpacked, dim=-1).reshape(out_f, -1)[:, :in_f]

    # INT4 is unsigned [0,15], center to signed [-8, 7]
    w_signed = w.float() - 8.0

    # Apply per-group scales
    w_grouped = w_signed.reshape(out_f, -1, group_size)
    scales = weight_scale.float().unsqueeze(-1)
    return (w_grouped * scales).reshape(out_f, -1)[:, :in_f].bfloat16()


def load_mtp_state_dict(mtp_weights_path, device="cpu"):
    """Load and dequantize MTP weights from safetensors.

    Returns a state dict with all weights in BF16, keyed by short names
    (without "model.layers.61." prefix).
    """
    log.info(f"Loading MTP weights from {mtp_weights_path}...")
    f = safe_open(mtp_weights_path, framework="pt", device=str(device))
    keys = list(f.keys())
    log.info(f"Total keys in mtp.safetensors: {len(keys)}")

    PREFIX = "model.layers.61."
    state_dict = {}
    expert_pattern = re.compile(
        r"mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight_packed|weight_scale|weight_shape)"
    )

    # Group expert quantized weights
    expert_parts = {}  # (expert_id, proj_name) -> {packed, scale, shape}

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
            # Non-quantized weight, store directly
            state_dict[short_key] = tensor

    # Dequantize expert weights
    log.info(f"Dequantizing {len(expert_parts)} expert projections...")
    for (expert_id, proj_name), parts in tqdm(
        sorted(expert_parts.items()), desc="Dequantizing experts"
    ):
        w = dequant_int4_gptq(
            parts["weight_packed"],
            parts["weight_scale"],
            parts["weight_shape"],
        )
        out_key = f"mlp.experts.{expert_id}.{proj_name}.weight"
        state_dict[out_key] = w

    log.info(f"State dict has {len(state_dict)} entries")
    return state_dict


def build_mtp_model(config_dir, state_dict, device):
    """Build standalone MTP layer from DeepSeek config + dequantized weights.

    Imports the actual DeepSeek model classes and instantiates a single decoder layer.
    """
    # Add config dir to path for importing modeling_deepseek
    config_dir = str(config_dir)
    if config_dir not in sys.path:
        sys.path.insert(0, config_dir)

    from configuration_deepseek import DeepseekV3Config
    from modeling_deepseek import DeepseekV3DecoderLayer, DeepseekV3RMSNorm

    # Load text config
    with open(Path(config_dir) / "config.json") as f:
        full_config = json.load(f)
    text_config = full_config.get("text_config", full_config)

    config = DeepseekV3Config(**text_config)
    config._attn_implementation = "eager"
    hidden_size = config.hidden_size  # 7168
    vocab_size = config.vocab_size  # 163840

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
            """
            Args:
                h_t: Hidden states from base model [batch, seq_len, hidden_size]
                x_next_ids: Next token IDs (input_ids shifted by 1) [batch, seq_len]
            Returns:
                logits: [batch, seq_len, vocab_size]
            """
            batch_size, seq_len, _ = h_t.shape

            # Embed next tokens and normalize
            x_embed = self.enorm(self.embed_tokens(x_next_ids))
            h_norm = self.hnorm(h_t)

            # Concatenate and project
            hidden = self.eh_proj(torch.cat([x_embed, h_norm], dim=-1))

            if debug:
                import logging as _log
                _l = _log.getLogger(__name__)
                _l.info("  h_t: mean=%.4f std=%.4f" % (h_t.float().mean().item(), h_t.float().std().item()))
                _l.info("  x_embed: mean=%.4f std=%.4f" % (x_embed.float().mean().item(), x_embed.float().std().item()))
                _l.info("  eh_proj out: mean=%.4f std=%.4f" % (hidden.float().mean().item(), hidden.float().std().item()))

            # Prepare position_ids for decoder layer
            position_ids = torch.arange(seq_len, device=h_t.device).unsqueeze(0).expand(batch_size, -1)

            # Create 4D causal attention mask [batch, 1, seq, seq]
            causal_mask = torch.full(
                (batch_size, 1, seq_len, seq_len), 
                torch.finfo(hidden.dtype).min, 
                device=hidden.device, dtype=hidden.dtype
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)
            # Expand to [batch, 1, seq, seq] - already correct shape

            # Run through decoder layer (attention + MoE MLP)
            layer_out = self.decoder_layer(
                hidden_states=hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
            )
            hidden = layer_out[0]

            if debug:
                _l.info("  decoder out: mean=%.4f std=%.4f" % (hidden.float().mean().item(), hidden.float().std().item()))

            # LM head
            logits = self.shared_head(self.shared_head_norm(hidden))
            return logits

    model = MTPLayer()

    # Map state dict keys to model parameters
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
        if src_key in key_mapping:
            dst_key = key_mapping[src_key]
            new_state[dst_key] = tensor
        elif src_key.startswith("self_attn."):
            dst_key = f"decoder_layer.{src_key}"
            new_state[dst_key] = tensor
        elif src_key.startswith("mlp."):
            dst_key = f"decoder_layer.{src_key}"
            new_state[dst_key] = tensor
        else:
            log.warning(f"Unmapped key: {src_key}")

    # Check for missing keys
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
        input_ids = sample["input_ids"]          # [S]
        hidden_states = sample["hidden_states"]  # [S, 7168]
        loss_mask = sample["loss_mask"]           # [S]

        seq_len = len(input_ids)
        if seq_len < 3:
            continue

        # MTP at position t: takes h_t + embed(x_{t+1}) -> predicts x_{t+2}
        # Valid range: t in [0, seq_len-3]
        # h_t: hidden_states[0:seq_len-2]
        # x_next: input_ids[1:seq_len-1]
        # target: input_ids[2:seq_len]
        # mask: loss_mask[2:seq_len] (only evaluate response positions)

        h_t = hidden_states[:-2].unsqueeze(0).to(device=device, dtype=torch.bfloat16)  # [1, S-2, D]
        x_next = input_ids[1:-1].unsqueeze(0).to(device=device)  # [1, S-2]
        targets = input_ids[2:].to(device=device)  # [S-2]
        mask = loss_mask[2:].to(device=device)  # [S-2]

        # Only process if there are response tokens
        if mask.sum() == 0:
            continue

        # Forward pass (full sequence, no chunking - attention needs full context)
        is_debug = len(per_sample_results) < 1
        logits = model(h_t, x_next, debug=is_debug).squeeze(0).float()  # [S-2, V]
        torch.cuda.empty_cache()

        # Top-1 accuracy
        preds = logits.argmax(dim=-1)  # [S-2]
        top1_match = (preds == targets) & (mask == 1)
        top1_correct = top1_match.sum().item()

        # Debug: print stats for first 3 samples
        if len(per_sample_results) < 3:
            log.info(
                "DEBUG sample %d: logits range=[%.2f, %.2f], std=%.4f, "
                "top1=%d/%d, pred[:5]=%s, target[:5]=%s",
                len(per_sample_results), logits.min().item(), logits.max().item(),
                logits.std().item(), top1_correct, mask.sum().item(),
                preds[:5].tolist(), targets[:5].tolist()
            )

        # Top-5 accuracy
        top5_preds = logits.topk(5, dim=-1).indices  # [S-2, 5]
        top5_match = (top5_preds == targets.unsqueeze(-1)).any(dim=-1) & (mask == 1)
        top5_correct = top5_match.sum().item()

        n_evaluated = mask.sum().item()
        total_top1_correct += top1_correct
        total_top5_correct += top5_correct
        total_evaluated += n_evaluated

        sample_result = {
            "file": pt_file.name,
            "seq_len": seq_len,
            "n_response_tokens": int(n_evaluated),
            "top1_correct": int(top1_correct),
            "top5_correct": int(top5_correct),
            "top1_rate": round(top1_correct / n_evaluated, 4) if n_evaluated > 0 else 0,
            "top5_rate": round(top5_correct / n_evaluated, 4) if n_evaluated > 0 else 0,
        }
        per_sample_results.append(sample_result)

    # Overall results
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


def main():
    args = parse_args()

    log.info("=" * 60)
    log.info("Phase 2: MTP Acceptance Rate Evaluation")
    log.info("=" * 60)

    # Step 1: Load and dequantize MTP weights
    state_dict = load_mtp_state_dict(args.mtp_weights, device="cpu")

    # Step 2: Build MTP model
    model = build_mtp_model(args.model_config, state_dict, args.device)
    del state_dict  # Free memory

    # Step 3: Evaluate
    results = evaluate_acceptance(model, args.data_dir, args.device, args.output)

    log.info("Phase 2 complete!")
    return results


if __name__ == "__main__":
    main()
