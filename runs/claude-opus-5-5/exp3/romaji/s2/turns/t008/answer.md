```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
def heads(l):
    lw=W[f'ln1_w.{l}']
    res=[]
    for h in range(6):
        s=slice(64*h,64*h+64)
        Q=W[f'W_q.{l}'][s]*lw; K=W[f'W_k.{l}'][s]*lw; V=W[f'W_v.{l}'][s]*lw; O=W[f'W_o.{l}'][:,s]
        res.append(dict(Q=Q,K=K,V=V,O=O,QK=Q.T@K/8,OV=O@V))
    return res
H=[heads(l) for l in range(4)]
for l in range(4):
  for h in range(6):
    A=H[l][h]['QK']
    S=(A+A.T)/2; Aa=(A-A.T)/2
    ev=np.linalg.eigvals(A)
    idx=np.argsort(-np.abs(ev))[:8]
    print(l,h,'sym %.3f anti %.3f'%(np.linalg.norm(S),np.linalg.norm(Aa)),'trace %.3f'%np.trace(A), 'top eig',np.round(ev[idx],3))
```