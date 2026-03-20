#!/usr/bin/env python3
"""
Compare K2.5 verifier hidden states (layer 60) extracted via:
1. Full-sequence prefill (FlashMLA)
2. Token-by-token decode (PagedAttention with MLA)

This tests whether the h_t in the eval dataset matches what vLLM
produces at each decode step during online serving.
"""
import os, sys, json, torch
import torch.nn.functional as F

os.environ["TORCHDYNAMO_DISABLE"] = "1"

# We use vLLM offline LLM with a hook on layer 60 to capture hidden states

def main():
    import torch
    from vllm import LLM, SamplingParams

    MODEL = "/data/models/Kimi-K2.5-MTP"
    DATA = "/data/datasets/apilog_k25_eagle3/val_5k_postnorm/data_0.pt"

    # Load data_0 (only 10 response tokens, clean for testing)
    d = torch.load(DATA, map_location="cpu", weights_only=True)
    input_ids = d["input_ids"]
    loss_mask = d["loss_mask"]
    dataset_hs = d["hidden_states"][-1]  # post-norm layer 60 hidden states [496, 7168]
    prompt_len = (loss_mask == 0).sum().item()
    total_len = len(input_ids)
    print(f"Sequence: {total_len} tokens (prompt={prompt_len}, response={total_len-prompt_len})")

    # Load K2.5 with TP=8
    print("Loading K2.5...")
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        tensor_parallel_size=8,
        max_model_len=4096,
        enforce_eager=True,
        dtype="bfloat16",
    )

    # Find layer 60 module
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    layer60 = None
    layer60_name = None
    for name, mod in model.named_modules():
        # Look for the 60th decoder layer
        if name.endswith(".layers.60") or name.endswith("[60]"):
            layer60 = mod
            layer60_name = name
            break
    if layer60 is None:
        # Try numeric access
        for name, mod in model.named_modules():
            if "layers" in name:
                try:
                    parts = name.split(".")
                    for i, p in enumerate(parts):
                        if p == "layers" and i + 1 < len(parts) and parts[i+1] == "60":
                            layer60 = mod
                            layer60_name = name
                            break
                except:
                    pass
            if layer60 is not None:
                break

    print(f"Layer 60 found: {layer60_name}, type={type(layer60).__name__ if layer60 else 'NOT FOUND'}")

    # List all modules with '60' to debug
    if layer60 is None:
        candidates = [(n, type(m).__name__) for n, m in model.named_modules()
                      if "60" in n.split(".")[-2:] if len(n.split(".")) > 1]
        print("Candidates:", candidates[:10])
        return

    captured = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        key = f"step_{captured.get('step', 0)}"
        captured[key] = hs.detach().cpu().clone()
        captured['step'] = captured.get('step', 0) + 1

    hook = layer60.register_forward_hook(hook_fn)

    # Method 1: Full prefill — send entire sequence
    print("\nRunning full-sequence prefill...")
    captured.clear()
    full_token_ids = input_ids.tolist()
    sampling = SamplingParams(max_tokens=1, temperature=0)
    from vllm import TokensPrompt
    llm.generate([TokensPrompt(prompt_token_ids=full_token_ids)],
                 sampling_params=sampling)

    prefill_captures = {k: v for k, v in captured.items() if k != 'step'}
    print(f"Prefill captured {len(prefill_captures)} forward passes")
    prefill_hs = list(prefill_captures.values())[0] if prefill_captures else None
    if prefill_hs is not None:
        print(f"  Shape: {prefill_hs.shape}")

    # Method 2: Prompt prefill + token-by-token decode
    print("\nRunning prompt prefill + decode...")
    captured.clear()
    prompt_ids = input_ids[:prompt_len].tolist()
    response_ids = input_ids[prompt_len:].tolist()

    # Prefill prompt
    llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)],
                 sampling_params=SamplingParams(max_tokens=1, temperature=0))
    # Note: We can't easily do token-by-token decode with the offline LLM API
    # So instead, we do incremental prefill: send prompt+resp[0], prompt+resp[0:2], etc.
    decode_hs_per_pos = {}
    for t in range(len(response_ids)):
        partial_ids = prompt_ids + response_ids[:t+1]
        captured.clear()
        llm.generate([TokensPrompt(prompt_token_ids=partial_ids)],
                     sampling_params=SamplingParams(max_tokens=1, temperature=0))
        caps = {k: v for k, v in captured.items() if k != 'step'}
        if caps:
            hs = list(caps.values())[0]
            # Last position in this sequence = position prompt_len + t
            pos = prompt_len + t
            decode_hs_per_pos[pos] = hs[..., pos, :] if hs.dim() >= 2 else hs

    hook.remove()

    # Compare
    if prefill_hs is not None and decode_hs_per_pos:
        print(f"\n=== Prefill vs incremental-prefill comparison ===")
        print(f"{'Pos':>5} {'Cosine':>10} {'MaxDiff':>10}")
        for pos in sorted(decode_hs_per_pos.keys()):
            if pos >= prefill_hs.shape[-2]:
                continue
            pf = prefill_hs[..., pos, :].float().view(-1)
            dc = decode_hs_per_pos[pos].float().view(-1)
            cos = F.cosine_similarity(pf.unsqueeze(0), dc.unsqueeze(0)).item()
            maxd = (pf - dc).abs().max().item()
            print(f"{pos:5d} {cos:10.6f} {maxd:10.4f}")

    # Compare with dataset h_t
    if prefill_hs is not None:
        print(f"\n=== Dataset h_t vs vLLM prefill h_t ===")
        for pos in range(min(total_len, prefill_hs.shape[-2])):
            ds = dataset_hs[pos].float()
            pf = prefill_hs[..., pos, :].float().view(-1)
            if ds.shape != pf.shape:
                continue
            cos = F.cosine_similarity(ds.unsqueeze(0), pf.unsqueeze(0)).item()
            if pos < 5 or pos >= prompt_len:
                print(f"pos {pos:3d}: cosine={cos:.6f}")

if __name__ == "__main__":
    main()
