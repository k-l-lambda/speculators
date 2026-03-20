#!/usr/bin/env python3
"""Compare MTP decoder sub-layer outputs (attn, MoE) with vLLM dump.
Tests with and without position-0 zeroing to measure its effect on each sub-layer.
"""
import os, sys, json, torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["TORCHDYNAMO_DISABLE"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts", "k2_mtp_config"))

from safetensors import safe_open
from configuration_deepseek import DeepseekV3Config
from modeling_deepseek import DeepseekV3Attention, DeepseekV3RMSNorm, DeepseekV3DecoderLayer
from speculators.train.data import shift_batch

MTP_PATH = "/data/models/Kimi-K2.5-MTP/mtp.safetensors"
CONFIG_PATH = os.path.join(ROOT_DIR, "scripts", "k2_mtp_config")
DATA_PATH = "/data/datasets/apilog_k25_eagle3/val_5k_postnorm/data_100.pt"
VLLM_ATTN = "/tmp/draft_dump/decoder_attn_out.pt"
VLLM_MOE = "/tmp/draft_dump/decoder_moe_out.pt"
VLLM_NORMED = "/tmp/draft_dump/decoder_normed.pt"
DEVICE = "cuda:0"
IDX = 100


def load_config():
    with open(os.path.join(CONFIG_PATH, "config.json")) as f:
        tc = json.load(f).get("text_config", {})
        if not tc:
            with open(os.path.join(CONFIG_PATH, "config.json")) as f2:
                tc = json.load(f2)
    config = DeepseekV3Config(**tc)
    config._attn_implementation = "eager"
    return config


def load_mtp_components(config):
    f = safe_open(MTP_PATH, framework="pt", device="cpu")
    prefix = "model.layers.61."
    H = config.hidden_size

    # Decoder layer
    decoder = DeepseekV3DecoderLayer(config, layer_idx=61)
    decoder_sd = {}
    for key in f.keys():
        if key.startswith(prefix) and not any(key.startswith(prefix + p) for p in ["enorm", "hnorm", "eh_proj", "embed_tokens", "shared_head"]):
            decoder_sd[key.removeprefix(prefix)] = f.get_tensor(key)
    decoder.load_state_dict(decoder_sd, strict=False)
    decoder = decoder.to(device=DEVICE, dtype=torch.bfloat16).eval()

    # enorm, hnorm, eh_proj, embed, input_layernorm
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

    return decoder, enorm, hnorm, eh_proj, embed


def prepare_input(config, enorm, hnorm, eh_proj, embed, zero_pos0=False):
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

    with torch.no_grad():
        x_embed = enorm(embed(x_ids))
        if zero_pos0:
            pos = torch.arange(x_ids.shape[1], device=DEVICE)
            x_embed = torch.where(pos.unsqueeze(0).unsqueeze(-1) == 0, torch.zeros_like(x_embed), x_embed)
        h_norm = hnorm(h_t)
        fused = eh_proj(torch.cat([x_embed, h_norm], dim=-1))

    return fused


def run_with_hooks(decoder, fused):
    """Run decoder layer and capture attn_out, moe_out via hooks."""
    captures = {}

    def hook_attn(module, input, output):
        # DeepseekV3Attention returns (attn_output, attn_weights, past_kv)
        captures["attn_out"] = output[0].detach().clone()

    def hook_mlp(module, input, output):
        captures["moe_out"] = output.detach().clone()

    h1 = decoder.self_attn.register_forward_hook(hook_attn)
    h2 = decoder.mlp.register_forward_hook(hook_mlp)

    bs, seq_len, _ = fused.shape
    position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0)
    causal_mask = torch.full(
        (bs, 1, seq_len, seq_len),
        torch.finfo(fused.dtype).min, device=DEVICE, dtype=fused.dtype,
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)

    with torch.no_grad():
        decoded = decoder(fused, attention_mask=causal_mask, position_ids=position_ids)

    h1.remove()
    h2.remove()

    return captures, decoded[0]  # decoded is (hidden_states, ...)


def compare(label, eval_tensor, vllm_tensor, positions):
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"{'Pos':>6} {'Cosine':>10} {'MaxDiff':>10} {'EvalNorm':>10} {'vLLMNorm':>10}")
    print("-" * 50)
    for idx in positions:
        ev = eval_tensor[idx].float()
        vl = vllm_tensor[idx].float()
        cos = F.cosine_similarity(ev.unsqueeze(0), vl.unsqueeze(0)).item()
        maxd = (ev - vl).abs().max().item()
        print(f"{idx:6d} {cos:10.6f} {maxd:10.4f} {ev.norm():10.2f} {vl.norm():10.2f}")

    # Average over all positions
    all_cos = []
    for i in range(min(len(eval_tensor), len(vllm_tensor))):
        c = F.cosine_similarity(eval_tensor[i:i+1].float(), vllm_tensor[i:i+1].float()).item()
        all_cos.append(c)
    print(f"\n  Average cosine (all {len(all_cos)} positions): {sum(all_cos)/len(all_cos):.6f}")
    print(f"  Min cosine: {min(all_cos):.6f} at pos {all_cos.index(min(all_cos))}")


def main():
    print("Loading config and model...")
    config = load_config()
    decoder, enorm, hnorm, eh_proj, embed = load_mtp_components(config)

    # Load vLLM dumps
    vllm_attn = torch.load(VLLM_ATTN, map_location="cpu", weights_only=True)
    vllm_moe = torch.load(VLLM_MOE, map_location="cpu", weights_only=True)
    vllm_normed = torch.load(VLLM_NORMED, map_location="cpu", weights_only=True)

    positions = [0, 1, 10, 50, 100, 150, 200, 219, 220]

    for zero_pos0 in [False, True]:
        tag = "WITH" if zero_pos0 else "WITHOUT"
        print(f"\n{'#'*60}")
        print(f"  {tag} position-0 zeroing")
        print(f"{'#'*60}")

        fused = prepare_input(config, enorm, hnorm, eh_proj, embed, zero_pos0=zero_pos0)
        captures, decoded = run_with_hooks(decoder, fused)

        # attn_out: hook captures the output of self_attn (before residual add)
        eval_attn = captures["attn_out"][0].cpu()
        eval_moe = captures["moe_out"][0].cpu()

        compare(f"Attention output ({tag} zeroing)", eval_attn, vllm_attn["attn_out"], positions)
        compare(f"MoE output ({tag} zeroing)", eval_moe, vllm_moe["moe_out"], positions)

        # Also compare normed (input_layernorm output = input to self_attn)
        # We need to compute normed from fused
        input_ln_weight = decoder.input_layernorm.weight.data
        with torch.no_grad():
            normed = decoder.input_layernorm(fused)
        eval_normed = normed[0].cpu()
        compare(f"Normed / input_layernorm ({tag} zeroing)", eval_normed, vllm_normed["normed"], positions)

    print("\nDone!")


if __name__ == "__main__":
    main()
