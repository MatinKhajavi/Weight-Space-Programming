```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=250)
hd=64
def hm(l,name,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_{name}.{l}'][s]
Up,Sp,_=np.linalg.svd(W['W_proj.0']); U=Up[:,:48]
special=[21,39,58,99,184,383]
for l in [1,2,3]:
    lnl=W[f'ln1_w.{l}']
    for j in range(6):
        Q=(hm(l,'q',j)*lnl).copy(); K=(hm(l,'k',j)*lnl).copy(); Q[:,special]=0; K[:,special]=0
        QK=Q.T@K/8
        B=U.T@QK@U
        sym=(B+B.T)/2; anti=(B-B.T)/2
        ev=np.linalg.eigvalsh(sym)
        print(f"L{l}H{j}: |B|={np.linalg.norm(B):.2f} |sym|={np.linalg.norm(sym):.2f} |anti|={np.linalg.norm(anti):.2f} eig(sym): {ev[:4]} ... {ev[-4:]}")
```