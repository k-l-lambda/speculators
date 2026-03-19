#!/usr/bin/env python3
"""Unit test: compare MLA attention between HF eager and FlashAttention.

Uses real MTP weights from mtp.safetensors and vLLM dump data as ground truth.
Tests whether FlashAttention produces outputs closer to vLLM than HF eager.

Usage:
    python tests/test_mla_attention.py
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts", "k2_mtp_config"))

from safetensors import safe_open
from configuration_deepseek import DeepseekV3Config
from modeling_deepseek import (
    DeepseekV3Attention,
    DeepseekV3RMSNorm,
    DeepseekV3DecoderLayer,
    apply_rotary_pos_emb,
)
from speculators.train.data import shift_batch


MTP_PATH = "/data/models/Kimi-K2.5-MTP/mtp.safetensors"
CONFIG_PATH = os.path.join(ROOT_DIR, "scripts", "k2_mtp_config")
DATA_PATH = "/data/datasets/apilog_k25_eagle3/val_5k_postnorm/data_100.pt"
VLLM_ATTN_DUMP = "/tmp/draft_dump/decoder_attn_out.pt"
DEVICE = "cuda:0"
IDX = 100  # position to compare


def load_config():
    with open(os.path.join(CONFIG_PATH, "config.json")) as f:
        tc = json.load(f).get("text_config", {})
        if not tc:
            with open(os.path.join(CONFIG_PATH, "config.json")) as f2:
                tc = json.load(f2)
    config = DeepseekV3Config(**tc)
    config._attn_implementation = "eager"
    return config


def prepare_fused_input(config):
    """Load data, build enorm/hnorm/eh_proj, compute fused tensor."""
    pt = torch.load(DATA_PATH, map_location="cpu", weights_only=True)
    ids = pt["input_ids"]
    hs = pt["hidden_states"]
    if isinstance(hs, list):
        hs = hs[-1]
    lm = pt["loss_mask"]
    if lm is not None and len(lm) != len(ids):
        lm = lm[:len(ids)]

    shifted = shift_batch({
        "input_ids": ids, "hidden_states": hs, "verifier_last_hidden_states": hs,
        "loss_mask": lm, "lengths": torch.tensor([len(ids)]),
        "position_ids": torch.arange(len(ids)),
    })

    h_t = shifted["hidden_states"].unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)
    x_ids = shifted["input_ids"].unsqueeze(0).to(DEVICE)

    H = config.hidden_size
    f = safe_open(MTP_PATH, framework="pt", device="cpu")
    prefix = "model.layers.61."

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

    with torch.no_grad():
        x_embed = enorm(embed(x_ids))
        h_norm = hnorm(h_t)
        fused = eh_proj(torch.cat([x_embed, h_norm], dim=-1))

    return fused


def run_hf_eager_attention(config, fused):
    """Run full decoder layer with HF eager attention."""
    import re

    def dequant_int4_gptq(wp, ws, wsh, gs=32):
        of, inf = wsh[0].item(), wsh[1].item()
        up = [(wp >> (i * 4)) & 0xF for i in range(8)]
        w = torch.stack(up, dim=-1).reshape(of, -1)[:, :inf]
        return ((w.float() - 8).reshape(of, -1, gs) * ws.float().unsqueeze(-1)).reshape(of, -1)[:, :inf].bfloat16()

    # Build attention module ONLY (no MoE — saves ~100GB)
    f = safe_open(MTP_PATH, framework="pt", device="cpu")
    prefix = "model.layers.61."

    attn = DeepseekV3Attention(config, layer_idx=61)
    input_ln = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # Load attention weights
    attn_sd = {}
    for key in f.keys():
        sk = key.removeprefix(prefix)
        if sk.startswith("self_attn."):
            attn_sd[sk.removeprefix("self_attn.")] = f.get_tensor(key)
    attn.load_state_dict(attn_sd, strict=False)
    input_ln.weight.data.copy_(f.get_tensor(prefix + "input_layernorm.weight"))

    attn = attn.to(device=DEVICE, dtype=torch.bfloat16).eval()
    input_ln = input_ln.to(device=DEVICE, dtype=torch.bfloat16).eval()

    bs, seq_len, _ = fused.shape
    position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0)
    causal_mask = torch.full(
        (bs, 1, seq_len, seq_len),
        torch.finfo(fused.dtype).min, device=DEVICE, dtype=fused.dtype,
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)

    with torch.no_grad():
        normed = input_ln(fused)
        attn_out, _, _ = attn(
            hidden_states=normed,
            attention_mask=causal_mask,
            position_ids=position_ids,
        )
        out = fused + attn_out

    # Wrap attn in a simple container for reuse
    class AttnContainer:
        pass
    container = AttnContainer()
    container.self_attn = attn
    container.input_layernorm = input_ln
    return out, container


def run_flash_attention_manual(config, fused, decoder):
    """Run MLA with FlashAttention instead of eager, using same weights."""
    from flash_attn import flash_attn_func

    attn = decoder.self_attn
    input_ln = decoder.input_layernorm

    bs, seq_len, _ = fused.shape
    position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        normed = input_ln(fused)

        # Q projection
        q = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(normed)))
        q = q.view(bs, seq_len, config.num_attention_heads, -1)
        q_nope, q_pe = q.split(
            [config.qk_nope_head_dim, config.qk_rope_head_dim], dim=-1
        )

        # KV projection
        compressed_kv = attn.kv_a_proj_with_mqa(normed)
        compressed_kv, k_pe = compressed_kv.split(
            [config.kv_lora_rank, config.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bs, seq_len, 1, config.qk_rope_head_dim)
        kv = attn.kv_b_proj(attn.kv_a_layernorm(compressed_kv)).view(
            bs, seq_len, config.num_attention_heads,
            config.qk_nope_head_dim + config.v_head_dim,
        )
        k_nope, v = kv.split([config.qk_nope_head_dim, config.v_head_dim], dim=-1)

        # RoPE — need to match HF's apply_rotary_pos_emb shapes
        # q_pe: [bs, seq, heads, rope_dim], k_pe: [bs, seq, 1, rope_dim]
        # apply_rotary_pos_emb expects [bs, heads, seq, dim]
        kv_seq_len = seq_len
        cos, sin = attn.rotary_emb(v.squeeze(2), seq_len=kv_seq_len)
        # Transpose to [bs, heads, seq, dim] for apply_rotary_pos_emb
        q_pe_t = q_pe.transpose(1, 2)  # [bs, heads, seq, rope_dim]
        k_pe_t = k_pe.transpose(1, 2)  # [bs, 1, seq, rope_dim]
        q_pe_t, k_pe_t = apply_rotary_pos_emb(q_pe_t, k_pe_t, cos, sin, position_ids)
        q_pe = q_pe_t.transpose(1, 2)  # back to [bs, seq, heads, rope_dim]
        k_pe = k_pe_t.transpose(1, 2)  # [bs, seq, 1, rope_dim]

        # Expand k_pe to all heads
        k_pe = k_pe.expand(-1, -1, config.num_attention_heads, -1)

        # Full Q and K
        q_full = torch.cat([q_nope, q_pe], dim=-1)
        k_full = torch.cat([k_nope, k_pe], dim=-1)

        # FlashAttention expects [bs, seq, heads, head_dim]
        # Q and K have qk_head_dim, V has v_head_dim
        # Need to pad V to qk_head_dim for flash_attn (or use separate softmax)
        # Actually flash_attn supports different head dims for Q/K vs V?
        # No - flash_attn requires same head_dim for Q, K, V

        # Pad V to match qk_head_dim
        qk_head_dim = q_full.shape[-1]
        v_padded = F.pad(v, [0, qk_head_dim - config.v_head_dim], value=0)

        # flash_attn_func(q, k, v, causal=True, softmax_scale=scale)
        softmax_scale = qk_head_dim ** -0.5
        flash_out = flash_attn_func(
            q_full, k_full, v_padded,
            causal=True,
            softmax_scale=softmax_scale,
        )
        # flash_out: [bs, seq, heads, qk_head_dim], take v_head_dim part
        flash_attn_output = flash_out[..., :config.v_head_dim]

        # o_proj
        flash_attn_output = flash_attn_output.reshape(bs, seq_len, -1)
        attn_output = attn.o_proj(flash_attn_output)

        # Residual add (same as HF decoder)
        after_attn = fused + attn_output

    return after_attn, flash_attn_output


def main():
    print("=" * 60)
    print("MLA Attention Comparison: HF Eager vs FlashAttention vs vLLM")
    print("=" * 60)

    config = load_config()
    print("\nPreparing fused input...")
    fused = prepare_fused_input(config)
    print(f"Fused shape: {fused.shape}")

    print("\nRunning HF eager attention...")
    hf_out, decoder = run_hf_eager_attention(config, fused)
    print(f"HF output shape: {hf_out.shape}")

    print("\nRunning FlashAttention...")
    flash_after_attn, flash_attn_raw = run_flash_attention_manual(config, fused, decoder)
    print(f"Flash output shape: {flash_after_attn.shape}")

    # Load vLLM reference
    if os.path.exists(VLLM_ATTN_DUMP):
        vllm_attn = torch.load(VLLM_ATTN_DUMP, map_location="cpu", weights_only=True)
        vllm_ref = vllm_attn["attn_out"][IDX].float()

        # Extract attention-only output (before residual add): attn_out = result - fused
        hf_attn_only = (hf_out[0, IDX] - fused[0, IDX]).cpu().float()
        flash_attn_only = (flash_after_attn[0, IDX] - fused[0, IDX]).cpu().float()

        cos_hf = F.cosine_similarity(hf_attn_only.unsqueeze(0), vllm_ref.unsqueeze(0)).item()
        cos_flash = F.cosine_similarity(flash_attn_only.unsqueeze(0), vllm_ref.unsqueeze(0)).item()

        print(f"\n{'=' * 60}")
        print(f"Comparison at index {IDX}:")
        print(f"  HF eager attn vs vLLM:     cosine = {cos_hf:.6f}")
        print(f"  FlashAttention vs vLLM:     cosine = {cos_flash:.6f}")
        print(f"  HF attn norm: {hf_attn_only.norm():.2f}, Flash attn norm: {flash_attn_only.norm():.2f}, vLLM norm: {vllm_ref.norm():.2f}")

        if cos_flash > cos_hf:
            print(f"\n  FlashAttention is CLOSER to vLLM by {cos_flash - cos_hf:.6f}")
        else:
            print(f"\n  HF eager is closer to vLLM (unexpected)")
    else:
        print(f"\nSKIP vLLM comparison: {VLLM_ATTN_DUMP} not found")


if __name__ == "__main__":
    main()
