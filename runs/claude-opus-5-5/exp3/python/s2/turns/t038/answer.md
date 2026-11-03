```python
import numpy as np
w0=W['ln1_w.0']; w2=W['ln2_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s[:k]
def cs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return u[:,:k]
def OV(l,j):
    w=W[f'ln1_w.{l}']; sl=slice(64*j,64*j+64); return W[f'W_o.{l}'][:,sl]@(W[f'W_v.{l}'][sl]*w)
alive=np.linalg.norm(W['W_fc.0'],axis=1)>1
F=(W['W_fc.0']*w2)[alive]
for kP in [16,24,32]:
  P,_=rs(np.vstack([W['W_q.0'][64:128]*w0, W['W_q.0'][320:384]*w0, W['W_k.0'][64:128]*w0, W['W_k.0'][320:384]*w0, W['W_v.0'][256:320]*w0,W['W_v.0'][0:64]*w0]),kP)
  for kO in [20,30]:
    Os=np.hstack([cs(OV(0,j),kO if j in (1,5) else 10) for j in range(6)])
    B=np.hstack([P,Os]); Q,_=np.linalg.qr(B)
    Fc=F-(F@Q)@Q.T
    s=np.linalg.svd(Fc,compute_uv=False)
    print(kP,kO,'F residual sv',np.round(s[:40],2))
```