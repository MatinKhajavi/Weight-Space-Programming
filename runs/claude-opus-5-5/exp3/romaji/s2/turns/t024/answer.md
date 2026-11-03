```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
spec=[5, 18, 21, 23, 29, 39, 58, 73, 76, 97, 99, 153, 168, 184, 214, 215, 236, 266, 275, 319, 322, 364, 375, 382, 383]
mask=np.ones(384,bool); mask[spec]=False; m=mask.sum()
writers={}
for l in range(4):
    lw=W[f'ln1_w.{l}']
    for h in range(6):
        s=slice(64*h,64*h+64)
        writers[f'L{l}H{h}']=(W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*lw)).astype(np.float64)
    writers[f'P{l}']=W[f'W_proj.{l}'].astype(np.float64)
def gram(names):
    M=np.zeros((m,m))
    for n in names:
        A=writers[n][mask]; M+=A@A.T/np.linalg.norm(A)**2
    return M
for grp in [['P3','P2','L3H0','L3H1','L3H2','L3H3','L3H5'],['P3','L3H3'],['P2','P3'],['P1','P2','P3'],list(writers)]:
    ev=np.linalg.eigvalsh(gram(grp))[::-1]
    print(grp if len(grp)<10 else 'all', (ev[:36]/ev[40:200].mean()).round(2))
```