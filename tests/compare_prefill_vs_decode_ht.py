#!/usr/bin/env python3
"""Compare hidden states from full-sequence prefill vs prompt-prefill + token-by-token decode.

This tests whether the verifier produces the same h_t in both modes,
which is the key question for understanding the eval-vLLM acceptance gap.
"""
import os, sys, json, torch
import torch.nn.functional as F

os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Use vLLM offline mode to load the full K2.5 model
from vllm import LLM, SamplingParams

MODEL = "/data/models/Kimi-K2.5-MTP"
DATA = "/data/datasets/apilog_k25_eagle3/val_5k_postnorm/data_100.pt"
DEVICE = "cuda"

def main():
    # Load dataset to get the token sequence
    d = torch.load(DATA, map_location="cpu", weights_only=True)
    input_ids = d["input_ids"]  # [496]
    loss_mask = d["loss_mask"]  # [496]
    dataset_hs = d["hidden_states"][-1]  # [496, 7168] - last layer, post-norm
    
    prompt_len = (loss_mask == 0).sum().item()
    total_len = len(input_ids)
    print(f"Sequence: {total_len} tokens (prompt={prompt_len}, response={total_len - prompt_len})")
    
    # Load K2.5 with vLLM (TP=8 for full model)
    print("Loading K2.5 model...")
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        tensor_parallel_size=8,
        max_model_len=4096,
        enforce_eager=True,
        dtype="bfloat16",
    )
    
    # Get the tokenizer
    tokenizer = llm.get_tokenizer()
    
    # Method 1: Full-sequence prefill
    # Send the entire sequence as a prompt and extract hidden states from layer 60
    prompt_token_ids = input_ids.tolist()
    
    # We need to monkey-patch the model to capture hidden states
    # Use a hook on the last decoder layer
    captured_hs = {}
    
    model_runner = llm.llm_engine.model_executor.driver_worker.model_runner
    model = model_runner.model
    
    # Find the last decoder layer (layer 60)
    # In KimiK25, the language model layers are under model.language_model.model.layers
    layers = None
    for name, mod in model.named_modules():
        if name.endswith(".layers") and hasattr(mod, "__len__"):
            layers = mod
            layer_name = name
            break
    
    if layers is None:
        # Try to find layers differently
        print("Searching for decoder layers...")
        for name, mod in model.named_modules():
            if "layers" in name and "60" in name:
                print(f"  Found: {name}: {type(mod).__name__}")
    
    print(f"Found layers at: {layer_name}, count={len(layers)}")
    
    # Hook on layer 60 output
    def hook_fn(module, input, output):
        # DecoderLayer returns (hidden_states, residual) or similar
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        captured_hs["layer60"] = hs.detach().clone()
    
    hook = layers[60].register_forward_hook(hook_fn)
    
    # Run prefill on the full sequence
    # Use generate with max_tokens=0 to just do prefill
    from vllm import TokensPrompt
    sampling = SamplingParams(max_tokens=1, temperature=0)
    result = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_token_ids)],
        sampling_params=sampling,
    )
    
    prefill_hs = captured_hs.get("layer60")
    hook.remove()
    
    if prefill_hs is not None:
        print(f"Prefill hidden states: {prefill_hs.shape}")
        
        # Compare with dataset
        dataset_hs_gpu = dataset_hs.to(prefill_hs.device, dtype=prefill_hs.dtype)
        
        # Compare at several positions
        print(f"\n=== Dataset h_t vs Full-prefill h_t ===")
        for idx in [0, 10, 50, 100, 200, 220, 250, 300, 400, 495]:
            if idx >= min(len(dataset_hs), prefill_hs.shape[-2]):
                break
            ds = dataset_hs_gpu[idx].float()
            pf = prefill_hs[0, idx].float() if prefill_hs.dim() == 3 else prefill_hs[idx].float()
            cos = F.cosine_similarity(ds.unsqueeze(0), pf.unsqueeze(0)).item()
            print(f"pos {idx:3d}: cosine={cos:.6f}, ds_norm={ds.norm():.2f}, pf_norm={pf.norm():.2f}")
    else:
        print("Failed to capture hidden states from prefill")

if __name__ == "__main__":
    main()
