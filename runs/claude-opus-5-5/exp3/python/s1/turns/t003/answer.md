```python
for nm in ['W_q.0','W_k.0','W_v.0','W_fc.0','W_o.0','W_proj.0','W_proj.3','W_o.3']:
    s=np.linalg.svd(W[nm],compute_uv=False)
    print(nm, s[10:70])
    print(s[180:240])
```