#!/usr/bin/env python3
"""Test MLA attention using SDPA vs HF eager, with identical bfloat16 weights.

Since all attention weights are bfloat16 (not INT4), the only difference
between eval and vLLM should be the attention computation path.
This test compares:
1. HF eager (matmul + softmax + matmul)
2. PyTorch SDPA (optimized, may use flash_attn under the hood)
3. vLLM dump ground truth
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
from modeling_deepseek import DeepseekV3RMSNorm, apply_rotary_pos_emb
from speculators.train.data import shift_batch

MTP_PATH = "/data/models/Kimi-K2.5-MTP/mtp.safetensors"
CONFIG_PATH = os.path.join(ROOT_DIR, "scripts", "k2_mtp_config")
DATA_PATH = "/data/datasets/apilog_k25_eagle3/val_5k_postnorm/data_100.pt"
VLLM_ATTN_DUMP = "/tmp/draft_dump/decoder_attn_out.pt"
DEVICE = "cuda:0"
IDX = 100


def load_config():
    with open(os.path.join(CONFIG_PATH, "config.json")) as f:
        tc = json.load(f).get("text_config", {})
        if not tc:
            with open(os.path.join(CONFIG_PATH, "config.json")) as f2:
                tc = json.load(f2)
    config = DeepseekV3Config(**tc)
    return config


class StandaloneMLA(nn.Module):
    """Standalone MLA attention using only bfloat16 weights + SDPA."""

    def __init__(self, config, device="cuda:0"):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.softmax_scale = self.qk_head_dim ** -0.5

        # Q projections
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = DeepseekV3RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        # KV projections
        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = DeepseekV3RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)

        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)

        # RoPE (reuse from HF)
        from modeling_deepseek import DeepseekV3RotaryEmbedding
        self.rotary_emb = DeepseekV3RotaryEmbedding(
            self.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward_eager(self, hidden_states, position_ids):
        """Standard HF-style eager attention."""
        bsz, q_len, _ = hidden_states.shape

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = compressed_kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv)).view(
            bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        ).transpose(1, 2)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        cos, sin = self.rotary_emb(v, seq_len=q_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        q_full = torch.cat([q_nope, q_pe], dim=-1)
        k_full = torch.cat([k_nope, k_pe.expand(-1, self.num_heads, -1, -1)], dim=-1)

        # Manual attention
        attn_weights = torch.matmul(q_full, k_full.transpose(2, 3)) * self.softmax_scale
        causal_mask = torch.triu(torch.full((q_len, q_len), float("-inf"), device=hidden_states.device), diagonal=1)
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_full.dtype)
        attn_output = torch.matmul(attn_weights, v)  # [bsz, heads, seq, v_dim]

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)

    def forward_sdpa(self, hidden_states, position_ids):
        """SDPA attention (uses flash_attn or efficient kernel under the hood)."""
        bsz, q_len, _ = hidden_states.shape

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = compressed_kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv)).view(
            bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        ).transpose(1, 2)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        cos, sin = self.rotary_emb(v, seq_len=q_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        q_full = torch.cat([q_nope, q_pe], dim=-1)
        k_full = torch.cat([k_nope, k_pe.expand(-1, self.num_heads, -1, -1)], dim=-1)

        # Pad V to match Q/K head dim for SDPA
        v_padded = F.pad(v, [0, self.qk_head_dim - self.v_head_dim])

        # SDPA (will use flash_attn if available)
        attn_output = F.scaled_dot_product_attention(
            q_full, k_full, v_padded,
            is_causal=True,
            scale=self.softmax_scale,
        )
        # Unpad V dimension
        attn_output = attn_output[..., :self.v_head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)


def main():
    print("=" * 60)
    print("MLA SDPA Test: same bf16 weights, different attention kernels")
    print("=" * 60)

    config = load_config()

    # Build standalone MLA
    mla = StandaloneMLA(config, DEVICE)

    # Load weights from mtp.safetensors
    f = safe_open(MTP_PATH, framework="pt", device="cpu")
    prefix = "model.layers.61.self_attn."
    sd = {}
    for key in f.keys():
        if key.startswith(prefix):
            sd[key.removeprefix(prefix)] = f.get_tensor(key)
    missing, unexpected = mla.load_state_dict(sd, strict=False)
    print(f"Loaded {len(sd) - len(missing)} attn weights, missing: {len(missing)}, unexpected: {len(unexpected)}")
    if missing:
        print(f"  Missing: {missing[:5]}")
    mla = mla.to(device=DEVICE, dtype=torch.bfloat16).eval()

    # Prepare input (fused after input_layernorm)
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

    h_t = shifted["hidden_states"].unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)
    x_ids = shifted["input_ids"].unsqueeze(0).to(DEVICE)

    # Build enorm/hnorm/eh_proj/input_layernorm
    H = config.hidden_size
    enorm = DeepseekV3RMSNorm(H, eps=config.rms_norm_eps)
    enorm.weight.data.copy_(f.get_tensor("model.layers.61.enorm.weight"))
    hnorm = DeepseekV3RMSNorm(H, eps=config.rms_norm_eps)
    hnorm.weight.data.copy_(f.get_tensor("model.layers.61.hnorm.weight"))
    eh_proj = nn.Linear(H * 2, H, bias=False)
    eh_proj.weight.data.copy_(f.get_tensor("model.layers.61.eh_proj.weight"))
    embed = nn.Embedding(config.vocab_size, H)
    embed.weight.data.copy_(f.get_tensor("model.layers.61.embed_tokens.weight"))
    input_ln = DeepseekV3RMSNorm(H, eps=config.rms_norm_eps)
    input_ln.weight.data.copy_(f.get_tensor("model.layers.61.input_layernorm.weight"))

    for m in [enorm, hnorm, eh_proj, embed, input_ln]:
        m.to(device=DEVICE, dtype=torch.bfloat16).eval()

    # Compute fused + input_layernorm
    bs, seq_len, _ = h_t.shape
    position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        x_embed = enorm(embed(x_ids))
        h_norm = hnorm(h_t)
        fused = eh_proj(torch.cat([x_embed, h_norm], dim=-1))
        normed = input_ln(fused)

        # Run both attention variants
        eager_out = mla.forward_eager(normed, position_ids)
        sdpa_out = mla.forward_sdpa(normed, position_ids)

    # Compare at IDX
    eager_v = eager_out[0, IDX].cpu().float()
    sdpa_v = sdpa_out[0, IDX].cpu().float()

    cos_eager_sdpa = F.cosine_similarity(eager_v.unsqueeze(0), sdpa_v.unsqueeze(0)).item()
    print(f"\nEager vs SDPA cosine at idx {IDX}: {cos_eager_sdpa:.6f}")

    if os.path.exists(VLLM_ATTN_DUMP):
        vllm_attn = torch.load(VLLM_ATTN_DUMP, map_location="cpu", weights_only=True)
        vllm_ref = vllm_attn["attn_out"][IDX].float()

        cos_eager_vllm = F.cosine_similarity(eager_v.unsqueeze(0), vllm_ref.unsqueeze(0)).item()
        cos_sdpa_vllm = F.cosine_similarity(sdpa_v.unsqueeze(0), vllm_ref.unsqueeze(0)).item()

        print(f"\nComparison at index {IDX}:")
        print(f"  Eager vs vLLM:  cosine = {cos_eager_vllm:.6f}")
        print(f"  SDPA vs vLLM:   cosine = {cos_sdpa_vllm:.6f}")
        print(f"  Eager vs SDPA:  cosine = {cos_eager_sdpa:.6f}")
        print(f"  Norms: eager={eager_v.norm():.2f}, sdpa={sdpa_v.norm():.2f}, vllm={vllm_ref.norm():.2f}")

        if cos_sdpa_vllm > cos_eager_vllm:
            print(f"\n  SDPA is CLOSER to vLLM by {cos_sdpa_vllm - cos_eager_vllm:.6f}")
        else:
            print(f"\n  Eager is closer (diff = {cos_eager_vllm - cos_sdpa_vllm:.6f})")
    else:
        print(f"\nNo vLLM dump at {VLLM_ATTN_DUMP}")


if __name__ == "__main__":
    main()
