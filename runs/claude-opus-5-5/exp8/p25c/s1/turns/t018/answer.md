```python
# Load activation arrays and align with samples
print(idx['layers'], idx['mlp_layers'], len(idx['samples']), idx['samples'][0])
H_mean = np.load(adir+'hidden_mean_answer.npy', mmap_mode='r'); H_last = np.load(adir+'hidden_last_prompt.npy', mmap_mode='r')
M_mean = np.load(adir+'mlp_mean_answer.npy', mmap_mode='r'); M_last = np.load(adir+'mlp_last_prompt.npy', mmap_mode='r')
print(H_mean.shape, H_mean.dtype, M_mean.shape, M_mean.dtype)
```