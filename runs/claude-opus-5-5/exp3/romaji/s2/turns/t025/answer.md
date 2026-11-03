```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
spec=[5, 18, 21, 23, 29, 39, 58, 73, 76, 97, 99, 153, 168, 184, 214, 215, 236, 266, 275, 319, 322, 364, 375, 382, 383]
mask=np.ones(384,bool); mask[spec]=False; m=mask.sum()
R={}
for l in range(4):
    R[f'fc{l}']=(W[f'W_fc.{l}']*W[f'ln2_w.{l}']).astype(np.float64)
    R[f'q{l}']=(W[f'W_q.{l}']*W[f'ln1_w.{l}']).astype(np.float64)
    R[f'k{l}']=(W[f'W_k.{l}']*W[f'ln1_w.{l}']).astype(np.float64)
    R[f'v{l}']=(W[f'W_v.{l}']*W[f'ln1_w.{l}']).astype(np.float64)
    R[f'fcraw{l}']=W[f'W_fc.{l}'].astype(np.float64)
def gram(names):
    M=np.zeros((m,m))
    for n in names:
        A=R[n][:,mask]; M+=A.T@A/np.linalg.norm(A)**2
    return M
for grp in [['fc0'],['fc1'],['fc2'],['fc3'],['fc1','fc2','fc3'],['fcraw1','fcraw2','fcraw3'],['v1','v2','v3'],['v0','fc0']]:
    ev=np.linalg.eigvalsh(gram(grp))[::-1]
    print(grp,(ev[:40]/ev[60:250].mean()).round(2))
```