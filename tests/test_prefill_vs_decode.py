"""Test MTP eval in decode mode (token-by-token with KV cache) vs prefill mode."""
import os, sys, torch
os.environ["TORCHDYNAMO_DISABLE"] = "1"
sys.path.insert(0, "/home/claude/work/speculators/src")

# Load greedy data_100
pt = torch.load("/data/datasets/apilog_k25_eagle3/val_5k_greedy/data_100.pt", map_location="cpu", weights_only=True)
ids = pt["input_ids"]
hs = pt["hidden_states"]
if isinstance(hs, list): hs = hs[-1]
lm = pt["loss_mask"]
if lm is not None and len(lm) != len(ids): lm = lm[:len(ids)]

from speculators.train.data import shift_batch
shifted = shift_batch({"input_ids": ids, "hidden_states": hs, "verifier_last_hidden_states": hs,
    "loss_mask": lm, "lengths": torch.tensor([len(ids)]), "position_ids": torch.arange(len(ids))})

h_t_full = shifted["hidden_states"].to("cuda:0", dtype=torch.bfloat16)  # [seq_len, hidden]
x_ids_full = shifted["input_ids"].to("cuda:0")  # [seq_len]
targets = torch.cat([shifted["input_ids"][1:], torch.zeros(1, dtype=shifted["input_ids"].dtype)]).to("cuda:0")
mask = shifted["loss_mask"].clone().to("cuda:0"); mask[-1] = 0

# Build model
exec_code = open("/home/claude/work/speculators/scripts/mtp_eval_acceptance.py").read()
exec_code = exec_code.split("def evaluate_acceptance")[0].split("@torch.no_grad")[0]
exec(exec_code)

sd = load_mtp_state_dict_k25("/data/models/Kimi-K2.5-MTP/mtp.safetensors", "cpu")
model = build_mtp_model("/home/claude/work/speculators/scripts/k2_mtp_config", sd, "cuda:0")
del sd

positions = [i for i in range(len(mask)) if mask[i] == 1]
seq_len = len(x_ids_full)

# === Prefill mode (standard) ===
with torch.no_grad():
    logits_prefill = model(h_t_full.unsqueeze(0), x_ids_full.unsqueeze(0)).squeeze(0).float()
preds_prefill = logits_prefill.argmax(dim=-1)
acc_prefill = sum(1 for p in positions if preds_prefill[p] == targets[p])
print("Prefill: %d/%d = %.4f" % (acc_prefill, len(positions), acc_prefill/len(positions)))

# === Decode mode (token-by-token with KV cache) ===
# Use transformers DynamicCache for KV cache
from transformers import DynamicCache

decode_preds = torch.zeros(seq_len, dtype=torch.long, device="cuda:0")
kv_cache = DynamicCache()

with torch.no_grad():
    for t in range(seq_len):
        h_t_t = h_t_full[t:t+1].unsqueeze(0)  # [1, 1, hidden]
        x_t = x_ids_full[t:t+1].unsqueeze(0)   # [1, 1]

        # MTP forward for single token
        xe = model.enorm(model.embed_tokens(x_t))
        hn = model.hnorm(h_t_t)
        hid = model.eh_proj(torch.cat([xe, hn], dim=-1))

        pos = torch.tensor([[t]], device="cuda:0")
        # Build causal mask for decode: [1, 1, 1, t+1] — attend to all past + current
        cm = torch.zeros((1, 1, 1, t + 1), device="cuda:0", dtype=hid.dtype)

        lo = model.decoder_layer(
            hidden_states=hid,
            attention_mask=cm,
            position_ids=pos,
            past_key_value=kv_cache,
            use_cache=True,
        )
        hid_out = lo[0]

        logits_t = model.shared_head(model.shared_head_norm(hid_out)).squeeze(0).squeeze(0).float()
        decode_preds[t] = logits_t.argmax(dim=-1)

acc_decode = sum(1 for p in positions if decode_preds[p] == targets[p])
print("Decode:  %d/%d = %.4f" % (acc_decode, len(positions), acc_decode/len(positions)))

# Per-position comparison for first 20 response tokens
print("\nPos  Prefill  Decode  Target  PfMatch DcMatch")
for p in positions[:20]:
    pf = preds_prefill[p].item()
    dc = decode_preds[p].item()
    tgt = targets[p].item()
    pm = "Y" if pf == tgt else "N"
    dm = "Y" if dc == tgt else "N"
    print("%4d %7d %7d %7d     %s       %s" % (p, pf, dc, tgt, pm, dm))
