```python
import numpy as np
w0=W['ln1_w.0']; 
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T
T=rs(np.vstack([W['W_v.0'][64:128]*w0, W['W_v.0'][320:384]*w0]),24)   # 384 x 24 token-ish subspace (raw residual coords)
def OV0(j):
    sl=slice(64*j,64*j+64); return W['W_o.0'][:,sl]@(W['W_v.0'][sl]*w0)
def QK(l,j):
    w=W[f'ln1_w.{l}']; sl=slice(64*j,64*j+64)
    return (W[f'W_q.{l}'][sl]*w).T@(W[f'W_k.{l}'][sl]*w)
def idscore(M):
    A=T.T@M@T; return np.trace(A)/np.linalg.norm(A), np.linalg.norm(A)
for l in [1,2,3]:
    for j in range(6):
        M=QK(l,j)
        d=idscore(M)
        row=[f'L{l}H{j} direct {d[0]:+.2f} ({d[1]:.2f})']
        for h in range(6):
            c=idscore(M@OV0(h)); row.append(f'h{h}:{c[0]:+.2f}({c[1]:.2f})')
        print(' '.join(row))
```