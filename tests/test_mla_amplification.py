#!/usr/bin/env python3
"""Test MLA attention amplification: how much does a small input perturbation
get amplified through the MLA attention computation?

Methodology:
1. Take real fused input from eval data
2. Add controlled noise at various levels (cosine 0.999, 0.998, 0.995, 0.99)
3. Run both original and perturbed through the SAME MLA attention (same weights)
4. Measure output cosine similarity
5. Compute amplification factor

This isolates the attention amplification from any weight or implementation difference.
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["TORCHDYNAMO_DISABLE"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts", "k2_mtp_config"))

from safetensors import safe_open
from configuration_deepseek import DeepseekV3Config
from modeling_deepseek import DeepseekV3Attention, DeepseekV3RMSNorm
from speculators.train.data import shift_batch

MTP_PATH = "/data/models/Kimi-K2.5-MTP/mtp.safetensors"
CONFIG_PATH = os.path.join(ROOT_DIR, "scripts", "k2_mtp_config")
DATA_PATH = "/data/datasets/apilog_k25_eagle3/val_5k_postnorm/data_100.pt"
DEVICE = "cuda:0"


def load_config():
    with open(os.path.join(CONFIG_PATH, "config.json")) as f:
        tc = json.load(f).get("text_config", {})
        if not tc:
            with open(os.path.join(CONFIG_PATH, "config.json")) as f2:
                tc = json.load(f2)
    config = DeepseekV3Config(**tc)
    config._attn_implementation = "eager"
    return config


def add_noise_to_cosine(x, target_cosine, seed=42):
    """Add noise to x such that cosine_similarity(x, x_noisy) ≈ target_cosine."""
    torch.manual_seed(seed)
    noise = torch.randn_like(x)
    # Project noise to be orthogonal to x, then mix
    # cos(x, x + α*noise_orth) = 1/sqrt(1 + α²)  when noise is orthogonal
    # We want: target_cosine = 1/sqrt(1 + α²)
    # → α² = 1/target_cosine² - 1
    # → α = sqrt(1/target_cosine² - 1)

    # Make noise orthogonal to x (per-position)
    # x: [batch, seq, hidden], noise: [batch, seq, hidden]
    dot = (x * noise).sum(dim=-1, keepdim=True)
    x_norm_sq = (x * x).sum(dim=-1, keepdim=True).clamp(min=1e-12)
    noise_orth = noise - dot / x_norm_sq * x
    # Normalize orthogonal noise to same norm as x
    noise_orth = noise_orth / noise_orth.norm(dim=-1, keepdim=True).clamp(min=1e-12) * x.norm(dim=-1, keepdim=True)

    alpha = (1.0 / target_cosine**2 - 1.0) ** 0.5
    x_noisy = x + alpha * noise_orth
    return x_noisy


def main():
    print("=" * 70)
    print("MLA Attention Amplification Test")
    print("=" * 70)

    config = load_config()

    # Load attention + layernorm
    f = safe_open(MTP_PATH, framework="pt", device="cpu")
    prefix = "model.layers.61."

    attn = DeepseekV3Attention(config, layer_idx=61)
    attn_sd = {}
    for key in f.keys():
        if key.startswith(prefix + "self_attn."):
            attn_sd[key.removeprefix(prefix + "self_attn.")] = f.get_tensor(key)
    attn.load_state_dict(attn_sd, strict=False)

    input_ln = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    input_ln.weight.data.copy_(f.get_tensor(prefix + "input_layernorm.weight"))

    attn = attn.to(device=DEVICE, dtype=torch.bfloat16).eval()
    input_ln = input_ln.to(device=DEVICE, dtype=torch.bfloat16).eval()

    # Prepare fused input
    pt = torch.load(DATA_PATH, map_location="cpu", weights_only=True)
    ids = pt["input_ids"]
    hs = pt["hidden_states"][-1] if isinstance(pt["hidden_states"], list) else pt["hidden_states"]
    lm = pt["loss_mask"]
    if lm is not None and len(lm) != len(ids):
        lm = lm[:len(ids)]
    shifted = shift_batch({
        "input_ids": ids, "hidden_states": hs, "verifier_last_hidden_states": hs,
        "loss_mask": lm, "lengths": torch.tensor([len(ids)]),
        "position_ids": torch.arange(len(ids)),
    })

    # Build fused
    H = config.hidden_size
    enorm = DeepseekV3RMSNorm(H, eps=config.rms_norm_eps)
    enorm.weight.data.copy_(f.get_tensor(prefix + "enorm.weight"))
    hnorm = DeepseekV3RMSNorm(H, eps=config.rms_norm_eps)
    hnorm.weight.data.copy_(f.get_tensor(prefix + "hnorm.weight"))
    eh_proj = nn.Linear(H * 2, H, bias=False)
    eh_proj.weight.data.copy_(f.get_tensor(prefix + "eh_proj.weight"))
    embed = nn.Embedding(config.vocab_size, H)
    embed.weight.data.copy_(f.get_tensor(prefix + "embed_tokens.weight"))
    for m in [enorm, hnorm, eh_proj, embed]:
        m.to(device=DEVICE, dtype=torch.bfloat16).eval()

    h_t = shifted["hidden_states"].unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)
    x_ids = shifted["input_ids"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        x_embed = enorm(embed(x_ids))
        h_norm = hnorm(h_t)
        fused = eh_proj(torch.cat([x_embed, h_norm], dim=-1))

    bs, seq_len, _ = fused.shape
    position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0)
    causal_mask = torch.full(
        (bs, 1, seq_len, seq_len),
        torch.finfo(fused.dtype).min, device=DEVICE, dtype=fused.dtype,
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)

    # Run clean attention
    with torch.no_grad():
        normed = input_ln(fused)
        clean_out, _, _ = attn(hidden_states=normed, attention_mask=causal_mask, position_ids=position_ids)

    # Test various noise levels
    target_cosines = [0.9999, 0.999, 0.998, 0.995, 0.99, 0.98, 0.95]

    print(f"\n{'Input cos':>12} {'Actual cos':>12} {'Output cos':>12} {'In diff':>10} {'Out diff':>10} {'Amplify':>10}")
    print("-" * 70)

    for target_cos in target_cosines:
        fused_noisy = add_noise_to_cosine(fused.float(), target_cos).to(dtype=torch.bfloat16)

        # Measure actual input cosine (per-position, then average)
        with torch.no_grad():
            normed_noisy = input_ln(fused_noisy)
            noisy_out, _, _ = attn(hidden_states=normed_noisy, attention_mask=causal_mask, position_ids=position_ids)

        # Compute cosines at multiple positions and average
        positions = list(range(50, 220, 10))
        in_cosines = []
        out_cosines = []
        for idx in positions:
            ic = F.cosine_similarity(
                normed[0, idx:idx+1].float(), normed_noisy[0, idx:idx+1].float()
            ).item()
            oc = F.cosine_similarity(
                clean_out[0, idx:idx+1].float(), noisy_out[0, idx:idx+1].float()
            ).item()
            in_cosines.append(ic)
            out_cosines.append(oc)

        avg_in_cos = sum(in_cosines) / len(in_cosines)
        avg_out_cos = sum(out_cosines) / len(out_cosines)
        in_diff = 1.0 - avg_in_cos
        out_diff = 1.0 - avg_out_cos
        amplify = out_diff / in_diff if in_diff > 1e-8 else float('inf')

        print(f"{target_cos:12.4f} {avg_in_cos:12.6f} {avg_out_cos:12.6f} {in_diff:10.6f} {out_diff:10.6f} {amplify:10.1f}x")

    # Also test with linear layer (no softmax) as baseline
    print(f"\n{'=' * 70}")
    print("Baseline: linear projection amplification (no softmax)")
    print(f"{'=' * 70}")

    linear = nn.Linear(config.hidden_size, config.hidden_size, bias=False).to(DEVICE, dtype=torch.bfloat16)
    nn.init.normal_(linear.weight, std=0.01)

    print(f"\n{'Input cos':>12} {'Output cos':>12} {'Amplify':>10}")
    print("-" * 40)

    for target_cos in [0.999, 0.998, 0.995, 0.99]:
        fused_noisy = add_noise_to_cosine(fused.float(), target_cos).to(dtype=torch.bfloat16)
        with torch.no_grad():
            lin_clean = linear(normed)
            lin_noisy = linear(input_ln(fused_noisy))

        cosines = []
        for idx in range(50, 220, 10):
            c = F.cosine_similarity(
                lin_clean[0, idx:idx+1].float(), lin_noisy[0, idx:idx+1].float()
            ).item()
            cosines.append(c)
        avg_cos = sum(cosines) / len(cosines)
        in_diff = 1 - target_cos
        out_diff = 1 - avg_cos
        amp = out_diff / in_diff if in_diff > 1e-8 else float('inf')
        print(f"{target_cos:12.4f} {avg_cos:12.6f} {amp:10.1f}x")


if __name__ == "__main__":
    main()
