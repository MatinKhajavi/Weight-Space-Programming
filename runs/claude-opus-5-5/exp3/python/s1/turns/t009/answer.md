```python
import numpy as np
d=384; H=6; hd=64
def head(l,j):
    s=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    Q=W[f'W_q.{l}'][s]*g; K=W[f'W_k.{l}'][s]*g; V=W[f'W_v.{l}'][s]*g; O=W[f'W_o.{l}'][:,s]
    return Q,K,V,O
def eigsum(M,k=20):
    e=np.linalg.eigvals(M)
    idx=np.argsort(-np.abs(e))[:k]
    e=e[idx]
    return e, e.real.sum()/np.abs(e).sum()
for l in range(4):
    for j in range(H):
        Q,K,V,O=head(l,j)
        e,r=eigsum(O@V,30)
        print(f'L{l}H{j} OV eig ratio {r:+.2f}', np.round(e[:5],2))
```