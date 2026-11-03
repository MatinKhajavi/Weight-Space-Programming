```python
import numpy as np
def QK(l,j):
    w=W[f'ln1_w.{l}']; sl=slice(64*j,64*j+64)
    return (W[f'W_q.{l}'][sl]*w).T@(W[f'W_k.{l}'][sl]*w)
def topplane(M):
    ev,V=np.linalg.eig(M); i=np.argmax(np.abs(ev))
    B=np.stack([V[:,i].real,V[:,i].imag],1); B,_=np.linalg.qr(B); return B, ev[i]
Bs=[topplane(QK(1,j))[0] for j in range(1,6)]
Pl,_s,_=np.linalg.svd(np.hstack(Bs),full_matrices=False); Pl=Pl[:,:_s.size][:, _s>0.5]
print('L1 plane union dim',Pl.shape[1], np.round(_s,2))
w0=W['ln1_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T
P=rs(np.vstack([W['W_q.0'][64:128]*w0, W['W_q.0'][320:384]*w0, W['W_k.0'][64:128]*w0, W['W_k.0'][320:384]*w0]),20)
for l in [1,2,3]:
    for j in range(6):
        M=QK(l,j); tot=np.linalg.norm(M)**2
        e=np.linalg.norm(Pl.T@M@Pl)**2/tot
        ep=np.linalg.norm(P.T@M@P)**2/tot
        # static key bias magnitude via ln1_b
        b=W[f'ln1_b.{l}']; w=W[f'ln1_w.{l}']; sl=slice(64*j,64*j+64)
        kpref=(W[f'W_k.{l}'][sl]*w).T@(W[f'W_q.{l}'][sl]@b)/8
        print(f'L{l}H{j} |QK| {np.sqrt(tot):5.1f}  energy in L1-pos-planes {e:.3f} in L0-P {ep:.3f}  static kpref {np.linalg.norm(kpref):.2f} (in planes {np.linalg.norm(Pl.T@kpref)/np.linalg.norm(kpref):.2f})')
```