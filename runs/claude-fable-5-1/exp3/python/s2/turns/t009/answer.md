```python
np.set_printoptions(precision=2, suppress=True, linewidth=200)
for nm in ['W_proj.3','W_o.3','W_proj.2','W_o.2','W_proj.1','W_o.1','W_proj.0','W_o.0']:
    s = np.linalg.svd(W[nm],compute_uv=False)
    print(nm, "sv:", s[:45])
```