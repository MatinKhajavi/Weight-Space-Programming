```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def headmats(l,j):
    g=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
s=slice(128,192)
Wq,Wk=W['W_q.0'][s],W['W_k.0'][s]
A2=Wq.T@Wk
u,sv,vt=np.linalg.svd(A2)
P=np.linalg.qr(np.concatenate([u[:,:14],vt[:14].T],1))[0]
g0=W['ln1_w.0']
Pr=P/np.maximum(g0,0.05)[:,None]
Pr=np.linalg.svd(Pr,full_matrices=False)[0]
d=Pr.shape[1]
for l in [1,2,3]:
    for j in range(6):
        Wq,Wk,Wv,Wo=headmats(l,j); A=Wq.T@Wk
        B=Pr.T@A@Pr
        fr=np.linalg.norm(B)**2/np.linalg.norm(A)**2/((d/384)**2)
        ev=np.linalg.eigvals(B); ev=ev[np.argsort(-np.abs(ev))][:6]
        print(f'L{l}H{j} PxP enrichment %.1f  |B| %.2f |A| %.2f'%(fr,np.linalg.norm(B),np.linalg.norm(A)),' top eig',[(round(abs(e),2),round(float(np.degrees(np.angle(e))))) for e in ev])
```