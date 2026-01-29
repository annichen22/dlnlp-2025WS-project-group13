import torch

def count_non_zero_params(model_path):
    state_dict = torch.load(model_path, map_location='cpu')
    total_params = 0
    non_zero_params = 0
    for name, param in state_dict.items():
        if 'weight' in name and 'LayerNorm' not in name:  # Focus on prunable weights
            total = param.numel()
            non_zero = (param != 0).sum().item()
            total_params += total
            non_zero_params += non_zero
            print(f'{name}: {non_zero}/{total} non-zero ({non_zero/total:.2%})')
    if total_params > 0:
        sparsity = 1 - (non_zero_params / total_params)
        print(f'Overall sparsity: {sparsity:.2%}')
    else:
        print('No prunable weights found.')

# Your paths
pruned_path = './temp_final_RTE_prun/pytorch_model.bin'
non_pruned_path = './data/final_tinybert_RTE_noDA_uniform/pytorch_model.bin'

print('Non-pruned:')
count_non_zero_params(non_pruned_path)

print('\nPruned:')
count_non_zero_params(pruned_path)