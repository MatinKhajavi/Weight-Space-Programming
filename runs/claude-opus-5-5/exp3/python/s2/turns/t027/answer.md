```python
import numpy as np
w0=W['ln1_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s[:k]
def QK(l,j):
    w=W[f'ln1_w.{l}']; sl=slice(64*j,64*j+64)
    return (W[f'W_q.{l}'][sl]*w).T@(W[f'W_k.{l}'][sl]*w)
T,_=rs(np.vstack([W['W_v.0'][64:128]*w0, W['W_v.0'][320:384]*w0]),27)
# H0 key token direction: top right sing vec of T-projected key side of H0 QK
M0=QK(0,0)
u,s,vt=np.linalg.svd(M0@T); h0k=vt[0]; print('H0 QK@T sv',np.round(s[:5],3))
u,s,vt=np.linalg.svd(T.T@M0); print('H0 T@QK sv (query token)',np.round(s[:5],3))
for l in [1]:
    for j in range(6):
        A=T.T@QK(l,j)@T; S=(A+A.T)/2
        ev,V=np.linalg.eigh(S)
        print(f'L{l}H{j} sym eig', np.round(ev[:6],2), '...', np.round(ev[-4:],2), ' |cos(most neg eigvec, H0key)|=%.2f'%abs(V[:,0]@h0k), ' diag-ish frac', round(np.trace(A)/np.abs(ev).sum(),2))
```