```python
import numpy as np
w0=W['ln1_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s[:k]
def QK(l,j):
    w=W[f'ln1_w.{l}']; sl=slice(64*j,64*j+64)
    return (W[f'W_q.{l}'][sl]*w).T@(W[f'W_k.{l}'][sl]*w)
def OV(l,j):
    w=W[f'ln1_w.{l}']; sl=slice(64*j,64*j+64); return W[f'W_o.{l}'][:,sl]@(W[f'W_v.{l}'][sl]*w)
T,_=rs(np.vstack([W['W_v.0'][64:128]*w0, W['W_v.0'][320:384]*w0]),40)
P,_=rs(np.vstack([W['W_q.0'][64:128]*w0, W['W_q.0'][320:384]*w0, W['W_k.0'][64:128]*w0, W['W_k.0'][320:384]*w0, W['W_v.0'][256:320]*w0,W['W_v.0'][0:64]*w0]),24)
Tp=T-P@(P.T@T); U,s,_=np.linalg.svd(Tp,full_matrices=False); Tp=U[:,:28]
def idsc(A):
    A=Tp.T@A@Tp; return np.trace(A)/np.linalg.norm(A)
print('max possible',np.sqrt(28).round(2))
for l in [1,2,3]:
    for j in range(6):
        M=QK(l,j)
        row=[f'L{l}H{j} direct {idsc(M):+.2f} |'] + [f'L0H{h}:{idsc(M@OV(0,h)):+.2f}' for h in [1,5]]
        if l>=2: row+=['|']+[f'L1H{h}:{idsc(M@OV(1,h)):+.2f}' for h in range(6)]
        if l>=3: row+=['|']+[f'L2H{h}:{idsc(M@OV(2,h)):+.2f}' for h in range(6)]
        print(' '.join(row))
print('OV copy-ness (trace on Tp of OV itself):')
for l in range(4):
    print(l,[round(idsc(OV(l,j)),2) for j in range(6)])
```