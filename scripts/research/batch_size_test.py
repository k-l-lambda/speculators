"""Test: does batch_size affect hidden states in VllmHiddenStatesGenerator?"""
import os, torch, torch.nn.functional as F
os.environ['TORCHDYNAMO_DISABLE'] = '1'

from speculators.data_generation.vllm_hidden_states_generator import VllmHiddenStatesGenerator

DATA_DIR = '/data/datasets/apilog_k25_eagle3/val_5k_postnorm'
MODEL = '/data/models/Kimi-K2.5-MTP'

d0 = torch.load(f'{DATA_DIR}/data_0.pt', map_location='cpu', weights_only=True)
ids0 = d0['input_ids'].tolist()
dataset_hs0 = d0['hidden_states'][-1]  # [1745, 7168]

d2 = torch.load(f'{DATA_DIR}/data_2.pt', map_location='cpu', weights_only=True)
ids2 = d2['input_ids'].tolist()  # 258 tokens

POSITIONS = [0, 10, 100, 500, 1000, 1500, 1734]

def extract(gen, seqs, label):
    print(f'\nExtracting {label} ({len(seqs)} seqs)...')
    results = gen.generate(seqs)
    r = results[0]
    if isinstance(r, dict) and 'hidden_states' in r:
        hs = r['hidden_states']
        if isinstance(hs, list):
            hs = hs[-1]
        return torch.tensor(hs) if not isinstance(hs, torch.Tensor) else hs
    return r

def compare(label, a, b):
    print(f'\n{label}:')
    for pos in POSITIONS:
        if pos >= len(a) or pos >= len(b): continue
        cos = F.cosine_similarity(a[pos:pos+1].float(), b[pos:pos+1].float()).item()
        print(f'  pos {pos:5d}: cosine={cos:.6f}')

# Generator 1: chunk=2048
print('=== Generator: max_num_batched_tokens=2048 ===')
gen = VllmHiddenStatesGenerator(model_path=MODEL, layer_ids=[60], max_model_len=4096,
                                 tensor_parallel_size=8, max_num_batched_tokens=2048)

hs_b1 = extract(gen, [ids0], 'data_0 alone (batch=1)')
hs_b2 = extract(gen, [ids0, ids2], 'data_0 + data_2 (batch=2)')

compare('batch=1 vs dataset', hs_b1, dataset_hs0)
compare('batch=2 data_0 vs dataset', hs_b2, dataset_hs0)
compare('batch=1 vs batch=2', hs_b1, hs_b2)

print('\n=== Done! ===')
