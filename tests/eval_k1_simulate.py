#!/usr/bin/env python3
"""Re-run eval on first 50 samples, get per-position predictions,
then simulate k=1 speculative decoding flow to get comparable acceptance rate."""
import os, sys, json, glob, torch
import torch.nn.functional as F
from tqdm import tqdm

os.environ['TORCHDYNAMO_DISABLE'] = '1'
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts/k2_mtp_config')

from safetensors import safe_open
from configuration_deepseek import DeepseekV3Config
from modeling_deepseek import DeepseekV3DecoderLayer, DeepseekV3RMSNorm
import torch.nn as nn
from speculators.train.data import shift_batch

MTP_PATH = '/data/models/Kimi-K2.5-MTP/mtp.safetensors'
CONFIG_PATH = 'scripts/k2_mtp_config'
DATA_DIR = '/data/datasets/apilog_k25_eagle3/val_5k_postnorm'
DEVICE = 'cuda:0'
MAX_SAMPLES = 50

def dequant(wp, ws, wsh, gs=32):
    of, inf = wsh[0].item(), wsh[1].item()
    u = [(wp >> (i*4)) & 0xF for i in range(8)]
    w = torch.stack(u, dim=-1).reshape(of, -1)[:, :inf]
    return ((w.float() - 8.0).reshape(of, -1, gs) * ws.float().unsqueeze(-1)).reshape(of, -1)[:, :inf].bfloat16()

def load_model():
    with open(os.path.join(CONFIG_PATH, 'config.json')) as f:
        tc = json.load(f).get('text_config', {})
    config = DeepseekV3Config(**tc)
    config._attn_implementation = 'eager'
    
    sf = safe_open(MTP_PATH, framework='pt', device='cpu')
    PREFIX = 'model.layers.61.'
    H = config.hidden_size
    
    enorm = DeepseekV3RMSNorm(H, eps=config.rms_norm_eps)
    hnorm = DeepseekV3RMSNorm(H, eps=config.rms_norm_eps)
    eh_proj = nn.Linear(H * 2, H, bias=False)
    embed = nn.Embedding(config.vocab_size, H)
    shared_head = nn.Linear(H, config.vocab_size, bias=False)
    
    enorm.weight.data.copy_(sf.get_tensor(PREFIX + 'enorm.weight'))
    hnorm.weight.data.copy_(sf.get_tensor(PREFIX + 'hnorm.weight'))
    eh_proj.weight.data.copy_(sf.get_tensor(PREFIX + 'eh_proj.weight'))
    embed.weight.data.copy_(sf.get_tensor(PREFIX + 'embed_tokens.weight'))
    norm_head = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    norm_head.weight.data.copy_(sf.get_tensor(PREFIX + 'shared_head.norm.weight'))
    shared_head.weight.data.copy_(sf.get_tensor(PREFIX + 'shared_head.head.weight'))
    
    state_dict = {}
    expert_parts = {}
    for key in sf.keys():
        if not key.startswith(PREFIX): continue
        short = key.removeprefix(PREFIX)
        tensor = sf.get_tensor(key)
        if any(short.startswith(p) for p in ['enorm', 'hnorm', 'eh_proj', 'embed_tokens', 'shared_head']):
            continue
        if '.experts.' in short:
            parts = short.split('.')
            eid, proj, part = int(parts[2]), parts[3], parts[4]
            ek = (eid, proj)
            if ek not in expert_parts: expert_parts[ek] = {}
            expert_parts[ek][part] = tensor
        else:
            state_dict[short] = tensor
    
    for (eid, pn), parts in sorted(expert_parts.items()):
        state_dict[f'mlp.experts.{eid}.{pn}.weight'] = dequant(parts['weight_packed'], parts['weight_scale'], parts['weight_shape'])
    
    decoder = DeepseekV3DecoderLayer(config, layer_idx=61)
    decoder.load_state_dict(state_dict, strict=False)
    
    for m in [enorm, hnorm, eh_proj, embed, shared_head, norm_head, decoder]:
        m.to(device=DEVICE, dtype=torch.bfloat16).eval()
    
    return config, enorm, hnorm, eh_proj, embed, shared_head, norm_head, decoder

def eval_sample(config, enorm, hnorm, eh_proj, embed, shared_head, norm_head, decoder, data_path):
    pt = torch.load(data_path, map_location='cpu', weights_only=True)
    ids = pt['input_ids']
    hs = pt['hidden_states'][-1] if isinstance(pt['hidden_states'], list) else pt['hidden_states']
    lm = pt['loss_mask']
    
    shifted = shift_batch({
        'input_ids': ids, 'hidden_states': hs, 'verifier_last_hidden_states': hs,
        'loss_mask': lm, 'lengths': torch.tensor([len(ids)]),
        'position_ids': torch.arange(len(ids)),
    })
    
    h_t = shifted['hidden_states'].unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)
    x_ids = shifted['input_ids'].unsqueeze(0).to(DEVICE)
    
    seq_len = h_t.shape[1]
    original_ids = ids.to(DEVICE)
    prompt_len = (lm == 0).sum().item()
    
    with torch.no_grad():
        x_embed = enorm(embed(x_ids))
        pos = torch.arange(x_ids.shape[1], device=DEVICE)
        x_embed = torch.where(pos.unsqueeze(0).unsqueeze(-1) == 0, torch.zeros_like(x_embed), x_embed)
        h_norm = hnorm(h_t)
        fused = eh_proj(torch.cat([x_embed, h_norm], dim=-1))
        
        causal_mask = torch.full((1, 1, seq_len, seq_len), torch.finfo(torch.bfloat16).min, device=DEVICE, dtype=torch.bfloat16)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0)
        
        decoded = decoder(fused, attention_mask=causal_mask, position_ids=position_ids)[0]
        logits = shared_head(norm_head(decoded))
        preds = logits[0].argmax(dim=-1)
    
    # eval position t: given (h[t], x[t+1]), predict x[t+2]
    # correct if preds[t] == original_ids[t+2]
    per_pos_correct = {}
    for t in range(seq_len):
        target_idx = t + 2
        if target_idx >= len(original_ids):
            break
        if t < prompt_len - 1:
            continue
        per_pos_correct[t] = (preds[t].item() == original_ids[target_idx].item())
    
    return per_pos_correct, prompt_len

def simulate_k1(lookup, prompt_len):
    """Simulate k=1 speculative decoding.
    
    eval_pos = P-1 means: MTP(h[P-1], x[P]) predicts x[P+1]
    
    k=1 flow after prefill (verifier generated x[P]):
    - First draft: eval_pos = P-1
    - If accept: bonus token x[P+2] generated, next eval_pos = P+1
    - If reject: verifier generates x[P+1], next eval_pos = P
    """
    if not lookup:
        return 0, 0
    
    max_pos = max(lookup.keys())
    total_drafted = 0
    total_accepted = 0
    
    eval_pos = prompt_len - 1
    
    while eval_pos <= max_pos:
        if eval_pos not in lookup:
            eval_pos += 1
            continue
        
        total_drafted += 1
        if lookup[eval_pos]:
            total_accepted += 1
            eval_pos += 2  # accepted: skip bonus token position
        else:
            eval_pos += 1  # rejected: next position
    
    return total_accepted, total_drafted

def main():
    print('Loading model...')
    config, enorm, hnorm, eh_proj, embed, shared_head, norm_head, decoder = load_model()
    
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'data_*.pt')))[:MAX_SAMPLES]
    print(f'Evaluating {len(files)} samples...')
    
    total_all = 0
    correct_all = 0
    total_k1_drafted = 0
    total_k1_accepted = 0
    
    for f in tqdm(files):
        lookup, prompt_len = eval_sample(config, enorm, hnorm, eh_proj, embed, shared_head, norm_head, decoder, f)
        
        for _, correct in lookup.items():
            total_all += 1
            if correct:
                correct_all += 1
        
        accepted, drafted = simulate_k1(lookup, prompt_len)
        total_k1_drafted += drafted
        total_k1_accepted += accepted
    
    print(f'\n=== Results ({len(files)} samples) ===')
    all_rate = correct_all / total_all if total_all else 0
    k1_rate = total_k1_accepted / total_k1_drafted if total_k1_drafted else 0
    print(f'All positions:  {correct_all}/{total_all} = {all_rate*100:.1f}%')
    print(f'Simulated k=1:  {total_k1_accepted}/{total_k1_drafted} = {k1_rate*100:.1f}%')
    print(f'vLLM k=1 online: 5499/8765 = 62.7%')
    print(f'Gap (sim vs vLLM): {(k1_rate - 5499/8765)*100:.1f}pp')

if __name__ == '__main__':
    main()
