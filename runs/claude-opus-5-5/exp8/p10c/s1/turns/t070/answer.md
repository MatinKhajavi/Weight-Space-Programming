The edit is stored the way the environment's `save_edit` intends: one `.npy` file named after the tensor (the original model is a single bf16 safetensors file with 320 tensors). So the file layout is probably not the cause, but I will verify the array's shape, dtype and finiteness to be sure. If it is a clean float32 array of shape (1024, 3584) with only column 998 zeroed, the repeated engine failures are most likely a server-side problem with starting the engines. In that case I will retry the same causal test, since failed requests are not counted.

```python
# Verify the saved .npy edit (shape/dtype/finite, only column 998 differs from original)
a = np.load('/work/edited/model.language_model.layers.16.mlp.down_proj.weight.npy')
o = load_original('model.language_model.layers.16.mlp.down_proj.weight').numpy()
diff_cols = np.where(np.abs(a - o).max(0) > 0)[0]
print(a.shape, a.dtype, 'finite', bool(np.isfinite(a).all()), 'changed cols', diff_cols.tolist(), 'max abs diff elsewhere', float(np.abs(np.delete(a,998,1)-np.delete(o,998,1)).max()))
```