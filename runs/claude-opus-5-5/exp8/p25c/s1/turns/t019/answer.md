The activation folder contains four arrays (residual stream and MLP inputs, each at the last prompt token and averaged over the answer) plus an index listing layers and samples. I now need to load them and align their rows with the generated samples and their grades, so I can compare finished-correct answers against looping ones. I expect the index to list sample ids in the same order as the array rows; if the shapes match 512 samples by the requested layers, I can proceed to a contrastive analysis. This step is purely bookkeeping, so there is no hypothesis to confirm yet beyond the format being usable.

```python
# Load activation arrays and align with samples
print(idx['layers'], idx['mlp_layers'], len(idx['samples']), idx['samples'][0])
H_mean = np.load(adir+'hidden_mean_answer.npy', mmap_mode='r'); H_last = np.load(adir+'hidden_last_prompt.npy', mmap_mode='r')
M_mean = np.load(adir+'mlp_mean_answer.npy', mmap_mode='r'); M_last = np.load(adir+'mlp_last_prompt.npy', mmap_mode='r')
print(H_mean.shape, H_mean.dtype, M_mean.shape, M_mean.dtype)
```