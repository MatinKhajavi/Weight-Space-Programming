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
U,S,Vt=np.linalg.svd(H[0][3]['OV'])
E=Vt[:20].T   # token-ish subspace (input side of L0H3 OV)
Eo=U[:,:20]
# raw (un-ln) residual token directions: input-side of OV with ln weights folded; the actual residual vector t satisfies (lw*t) in E. so t ~ E/lw
lw0=W['ln1_w.0']
T=E/lw0[:,None]; T,_=np.linalg.qr(T)
OV3=W['W_o.0'][:,192:256]@W['W_v.0'][192:256]*lw0  # maps raw residual to output
for l in [1,2,3]:
    for h in range(6):
        QK=H[l][h]['QK']
        M=T.T@QK@OV3@T
        ev=np.linalg.eigvals(M)
        M2=T.T@QK@T   # same token matching
        print(l,h,'ind: tr/fro %.3f fro %.3f'%(np.trace(M)/np.linalg.norm(M),np.linalg.norm(M)),'pos-real-eig-frac %.2f'%(ev.real.sum()/np.abs(ev).sum()),
              '| same-tok tr/fro %.3f fro %.3f'%(np.trace(M2)/np.linalg.norm(M2),np.linalg.norm(M2)))
```