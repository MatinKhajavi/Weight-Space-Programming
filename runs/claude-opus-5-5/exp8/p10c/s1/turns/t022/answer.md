```python
# Hypothesis: load activation arrays and align with samples.
print(idx['layers'], idx['mlp_layers'])
print(idx['samples'][:2])
H_mean = np.load(ad+'hidden_mean_answer.npy', mmap_mode='r'); H_last = np.load(ad+'hidden_last_prompt.npy', mmap_mode='r')
M_mean = np.load(ad+'mlp_mean_answer.npy', mmap_mode='r'); M_last = np.load(ad+'mlp_last_prompt.npy', mmap_mode='r')
print(H_mean.shape, H_mean.dtype, H_last.shape, M_mean.shape, M_mean.dtype, M_last.shape)
```