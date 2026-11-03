```python
import numpy as np
w0=W['ln1_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s[:k]
def cs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return u[:,:k]
def QK(l,j):
    w=W[f'ln1_w.{l}']; sl=slice(64*j,64*j+64)
    return (W[f'W_q.{l}'][sl]*w).T@(W[f'W_k.{l}'][sl]*w)
def OV0(j):
    sl=slice(64*j,64*j+64); return W['W_o.0'][:,sl]@(W['W_v.0'][sl]*w0)
O4=cs(np.hstack([OV0(4),OV0(0)]),3)
P,_=rs(np.vstack([W['W_q.0'][64:128]*w0, W['W_q.0'][320:384]*w0, W['W_k.0'][64:128]*w0, W['W_k.0'][320:384]*w0]),20)
Mo=cs(W['W_proj.0'],60)
for l in [1,2]:
    for j in range(6):
        M=QK(l,j); ev,V=np.linalg.eig(M); i=np.argmax(np.abs(ev))
        if ev[i].imag<0: i=np.argmin(np.abs(ev-np.conj(ev[i])))
        B=np.stack([V[:,i].real,V[:,i].imag],1); B,_=np.linalg.qr(B)
        print(f'L{l}H{j} top eig {abs(ev[i]):.2f}@{np.degrees(np.angle(ev[i])):.0f}  plane in O4 {np.linalg.norm(O4.T@B)**2/2:.2f}  in P {np.linalg.norm(P.T@B)**2/2:.2f} in MLP0out60 {np.linalg.norm(Mo.T@B)**2/2:.2f}')
print('O4 sv', np.round(np.linalg.svd(np.hstack([OV0(4),OV0(0)]),compute_uv=False)[:5],2))
# overlap of O4 with P and with raw coordinates
print('O4 vs P',np.round(np.linalg.svd(O4.T@P,compute_uv=False),2))
```