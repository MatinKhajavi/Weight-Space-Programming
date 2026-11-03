```python
import numpy as np
mass=[222,78,72,360,103,254,25,249,226,124,327,197,273]
def spec(M,k=34):
    M=M.copy(); 
    return np.round(np.linalg.svd(M,compute_uv=False)[:k],2)
P3=W['W_proj.3']
# remove basis-aligned massive rows to see remaining spectrum
mask=np.ones(384,bool); mask[mass]=False
print('proj3 no-massive', spec(P3[mask]))
OV3=np.hstack([W['W_o.3'][:,64*h:64*h+64]@(W['W_v.3'][64*h:64*h+64]*W['ln1_w.3']) for h in range(6)])
print('L3 OV concat no-massive', spec(OV3[mask]))
print('proj2 no-massive', spec(W['W_proj.2'][mask]))
# combined final writers normalized
C=np.hstack([P3[mask]/np.linalg.norm(P3), OV3[mask]/np.linalg.norm(OV3)])
print('combined', spec(C))
```