```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def heads(l):
    lw=W[f'ln1_w.{l}']
    res=[]
    for h in range(6):
        s=slice(64*h,64*h+64)
        Q=W[f'W_q.{l}'][s]*lw; K=W[f'W_k.{l}'][s]*lw; V=W[f'W_v.{l}'][s]*lw; O=W[f'W_o.{l}'][:,s]
        res.append(dict(Q=Q,K=K,V=V,O=O,QK=Q.T@K/8,OV=O@V))
    return res
H=[heads(l) for l in range(4)]
def sv(A): return np.linalg.svd(A)
lw0=W['ln1_w.0']
U,S,Vt=sv(H[0][3]['QK'])
Pos=np.linalg.qr(np.hstack([U[:,:16],Vt[:16].T])/lw0[:,None])[0][:,:20]
B0=Pos.T@H[0][3]['QK']@Pos
# rescale to raw: L0 QK acts on lw-scaled; compute raw version: QK_raw = diag(lw)QKdiag(lw) already folded since Q includes lw. Yes H contains lw.
def cs(a,b): return (a*b).sum()/np.linalg.norm(a)/np.linalg.norm(b)
for l in range(4):
    for h in range(6):
        B=Pos.T@H[l][h]['QK']@Pos
        ev=np.linalg.eigvals(B); idx=np.argsort(-np.abs(ev))[:6]
        print(l,h,'|B| %.3f'%np.linalg.norm(B),'cos with L0H3 %.2f'%cs(B,B0),'anti/sym %.2f'%(np.linalg.norm(B-B.T)/np.linalg.norm(B+B.T)),'tr %.3f'%np.trace(B),'eig',np.round(ev[idx],3))
```