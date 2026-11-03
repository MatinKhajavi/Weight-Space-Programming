```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def headmats(l,j):
    g=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
def pr(M):
    ev=np.linalg.eigvals(M); return ev.real.sum()/np.abs(ev).sum()
Wq,Wk,Wv,Wo=headmats(0,2); OV2=Wo@Wv
for l in [1,2,3]:
    for j in range(6):
        Wq,Wk,Wv,Wo=headmats(l,j); A=Wq.T@Wk
        print(f'L{l}H{j}: self %.2f | q=id,k=prev %.2f | q=prev,k=prev %.2f | q=prev,k=id %.2f'%(pr(A),pr(A@OV2),pr(OV2.T@A@OV2),pr(OV2.T@A)))
```