```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
spec=[5, 18, 21, 23, 29, 39, 58, 73, 76, 97, 99, 153, 168, 184, 214, 215, 236, 266, 275, 319, 322, 364, 375, 382, 383]
mask=np.ones(384,bool); mask[spec]=False; m=mask.sum()
P3=W['W_proj.3'].astype(np.float64)[mask]
U,S,Vt=np.linalg.svd(P3)
lw=W['ln1_w.3']; s=slice(192,256)
OV=(W['W_o.3'][:,s]@(W['W_v.3'][s]*lw))[mask]
Uo,So,Vo=np.linalg.svd(OV)
print('L3H3 sv',So[:36].round(2))
for kk in [21,24,27,30]:
    print('frac of each L3H3 out dir in P3 top',kk,(np.linalg.norm(U[:,:kk].T@Uo[:,:36],axis=0)**2).round(2))
print('random expectation for 24:',24/m)
```