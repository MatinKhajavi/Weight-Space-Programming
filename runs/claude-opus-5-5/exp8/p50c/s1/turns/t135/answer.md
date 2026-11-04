The attention maps are stored as float16. I'll check the dtype of the hidden-state and MLP-activation files to finalize the budget.

```python
# Hypothesis: hidden/mlp arrays are float16 too (halves the per-token size)
for f in ['hidden_0.npy','hidden_24.npy','mlp_act_22.npy','topk_logprobs.npy','token_logprobs.npy']:
    a=np.load('/work/fwd_11/'+f, mmap_mode='r'); print(f, a.shape, a.dtype)
```