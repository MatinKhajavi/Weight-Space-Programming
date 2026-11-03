```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
sink=[21,39,58,99,184]
for name in ['W_proj.3','W_o.3','W_proj.2']:
    U,S,Vt=np.linalg.svd(W[name])
    print(name)
    print('  sv     ', S[:36])
    print('  sinkfrac', np.linalg.norm(U[sink,:36],axis=0)**2)
# also remove sink dims and redo
M=W['W_proj.3'].copy(); mask=np.ones(384,bool); mask[sink]=False
print('W_proj.3 without sink rows', np.linalg.svd(M[mask],compute_uv=False)[:40])
print('W_o.3 without sink rows', np.linalg.svd(W['W_o.3'][mask],compute_uv=False)[:40])
```