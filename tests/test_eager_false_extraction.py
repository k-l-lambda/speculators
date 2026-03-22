#!/usr/bin/env python3
"""Test VllmHiddenStatesGenerator with enforce_eager=False."""
import os, sys
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
sys.path.insert(0, 'src')

import torch
import torch.nn.functional as F


def main():
    from speculators.data_generation.vllm_hidden_states_generator import VllmHiddenStatesGenerator
    import speculators.data_generation.vllm_hidden_states_generator as vhsg

    _orig = vhsg.VllmHiddenStatesGenerator._create_vllm_config
    def _patched(self, *args, **kwargs):
        cfg = _orig(self, *args, **kwargs)
        cfg.model_config.enforce_eager = False
        print(f'[patched] enforce_eager={cfg.model_config.enforce_eager}', flush=True)
        return cfg
    vhsg.VllmHiddenStatesGenerator._create_vllm_config = _patched

    print('Creating generator with enforce_eager=False...')
    gen = VllmHiddenStatesGenerator(
        model_path='/data/models/Kimi-K2.5-MTP', layer_ids=[60],
        max_model_len=4096, tensor_parallel_size=8, max_num_batched_tokens=8192,
    )
    print('Generator ready. Extracting data_0...')

    d = torch.load('/data/datasets/apilog_k25_eagle3/val_5k_v2/data_0.pt',
                   map_location='cpu', weights_only=True)
    dataset_hs = d['hidden_states'][-1]

    results = gen.generate([d['input_ids'].tolist()])
    r = results[0]
    hs_list = r.get('hidden_states')
    hs = hs_list[-1] if isinstance(hs_list, list) else hs_list
    if not isinstance(hs, torch.Tensor):
        hs = torch.tensor(hs)
    print(f'Extracted h_t shape: {hs.shape}')

    n = min(len(hs), len(dataset_hs), 20)
    cos_list = [F.cosine_similarity(hs[i:i+1].float(), dataset_hs[i:i+1].float()).item()
                for i in range(n)]
    avg = sum(cos_list) / len(cos_list)
    print(f'Cosine vs dataset (eager=True) first {n} positions: avg={avg:.6f}')
    print(f'  {[round(c,4) for c in cos_list[:10]]}')

    torch.save({'hidden_states': [hs], 'input_ids': d['input_ids'], 'loss_mask': d['loss_mask']},
               '/tmp/data_0_eager_false.pt')
    print('Saved /tmp/data_0_eager_false.pt')


if __name__ == '__main__':
    main()
