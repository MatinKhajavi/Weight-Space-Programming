```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
spec=[5, 18, 21, 23, 29, 39, 58, 73, 76, 97, 99, 153, 168, 184, 214, 215, 236, 266, 275, 319, 322, 364, 375, 382, 383]
mask=np.ones(384,bool); mask[spec]=False; m=mask.sum()
P3=W['W_proj.3'].astype(np.float64)[mask]
U,S,Vt=np.linalg.svd(P3)
def enr(A):
    A=A[mask]; return ((U.T@A)**2).sum(1)/np.linalg.norm(A)**2*m
rows={}
for l in [2,3]:
    lw=W[f'ln1_w.{l}']
    for h in range(6):
        s=slice(64*h,64*h+64)
        rows[f'L{l}H{h}']=enr(W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*lw))
rows['P2']=enr(W['W_proj.2']); rows['P1']=enr(W['W_proj.1']); rows['P0']=enr(W['W_proj.0'])
print('dir  ',np.arange(34))
print('sv   ',S[:34].round(1))
for k,v in rows.items(): print(k.ljust(5),v[:34].round(1))
```