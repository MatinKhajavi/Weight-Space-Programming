```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
spec=[5, 18, 21, 23, 29, 39, 58, 73, 76, 97, 99, 153, 168, 184, 214, 215, 236, 266, 275, 319, 322, 364, 375, 382, 383]
mask=np.ones(384,bool); mask[spec]=False
def U_(n,k): return np.linalg.svd(W[n].astype(np.float64)[mask])[0][:,:k]
for a,b in [('W_proj.3','W_proj.2'),('W_proj.3','W_proj.1'),('W_proj.2','W_proj.1'),('W_proj.3','W_proj.0'),('W_proj.2','W_proj.0')]:
    for k in [30]:
        c=np.linalg.svd(U_(a,k).T@U_(b,k),compute_uv=False)
        print(a,b,k,c.round(2))
# random baseline
rng=np.random.default_rng(1); m=mask.sum()
A=np.linalg.qr(rng.standard_normal((m,30)))[0]; B=np.linalg.qr(rng.standard_normal((m,30)))[0]
print('random',np.linalg.svd(A.T@B,compute_uv=False).round(2))
```