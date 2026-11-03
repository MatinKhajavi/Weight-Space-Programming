```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
for name in ['W_proj.3','W_o.3','W_proj.2','W_o.2','W_proj.1','W_proj.0']:
    s=np.linalg.svd(W[name],compute_uv=False)
    print(name, s[:60])
    print('   ...', s[60:120:4])
```