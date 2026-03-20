import os, sys
os.environ['TORCHDYNAMO_DISABLE'] = '1'
sys.path.insert(0, 'src')

def main():
    from speculators.data_generation.vllm_hidden_states_generator import VllmHiddenStatesGenerator
    import torch
    import torch.nn.functional as F

    new_data = torch.load('/tmp/data_2_eager_tokens.pt', map_location='cpu', weights_only=True)
    new_ids = new_data['input_ids'].tolist()
    prompt_len = (new_data['loss_mask'] == 0).sum().item()
    print(f'Sequence: {len(new_ids)} tokens, prompt={prompt_len}')

    print('Extracting h_t (enforce_eager=True)...')
    gen = VllmHiddenStatesGenerator(
        model_path='/data/models/Kimi-K2.5-MTP', layer_ids=[60], max_model_len=4096,
        tensor_parallel_size=8, max_num_batched_tokens=8192,
    )
    results = gen.generate([new_ids])
    r = results[0]
    new_hs = r['hidden_states'][-1] if isinstance(r['hidden_states'], list) else r['hidden_states']
    if not isinstance(new_hs, torch.Tensor):
        new_hs = torch.tensor(new_hs)
    print(f'Extracted h_t: {new_hs.shape}')

    torch.save({
        'input_ids': new_data['input_ids'],
        'hidden_states': [new_hs],
        'loss_mask': new_data['loss_mask'],
    }, '/tmp/data_2_consistent.pt')

    old = torch.load('/data/datasets/apilog_k25_eagle3/val_5k_postnorm/data_2.pt', map_location='cpu', weights_only=True)
    old_hs = old['hidden_states'][-1]
    n = min(prompt_len, len(new_hs), len(old_hs))
    cos_list = [F.cosine_similarity(new_hs[i:i+1].float(), old_hs[i:i+1].float()).item() for i in range(n)]
    print(f'Prompt avg cosine (new vs old dataset): {sum(cos_list)/len(cos_list):.6f}, min={min(cos_list):.6f}')
    print('Done!')

if __name__ == '__main__':
    main()
