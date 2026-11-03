```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=250)
def headmats(l,h):
    lw=W[f'ln1_w.{l}']; s=slice(64*h,64*h+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
lw0=W['ln1_w.0']
ins=[]
for h in [1,2,4,5]:
    q,k,v,o=headmats(0,h); ins.append(np.linalg.svd(o@v)[2][:2].T)
PinL=np.linalg.svd(np.hstack(ins))[0][:,:2]     # in coordinates of normalized residual (pre-lw)?? note: v includes lw so row space is in normalized-residual coords
for l in range(4):
    for h in range(6):
        q,k,v,o=headmats(l,h)
        if l==0: B=PinL.T@(q.T@k/8)@PinL
        else:
            Pr=np.linalg.qr(PinL)[0]; B=Pr.T@(q.T@k/8)@Pr
        print(l,h,B.round(3).tolist(), 'eig',np.round(np.linalg.eigvals(B),3))
```