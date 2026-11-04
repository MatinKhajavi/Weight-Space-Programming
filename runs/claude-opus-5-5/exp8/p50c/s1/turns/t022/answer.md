```python
# Hypothesis: load activation arrays; check shapes and align with per-sample cut/loop-onset info.
H_last = np.load('/work/gen_3_acts/hidden_last_prompt.npy', mmap_mode='r')
H_mean = np.load('/work/gen_3_acts/hidden_mean_answer.npy', mmap_mode='r')
M_last = np.load('/work/gen_3_acts/mlp_last_prompt.npy', mmap_mode='r')
M_mean = np.load('/work/gen_3_acts/mlp_mean_answer.npy', mmap_mode='r')
print(H_last.shape, H_mean.shape, M_last.shape, M_mean.shape, H_mean.dtype)
samples = idx['samples']
print(len(samples), sum(s['correct'] for s in samples))
```