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
for l1 in range(2):
  for j1 in range(H):
    Q1,K1,V1,O1=head(l1,j1)
    OV=O1@V1
    out=[]
    for l2 in range(l1+1,4):
        for j2 in range(H):
            Q2,K2,V2,O2=head(l2,j2)
            M=Q2.T@K2@OV
            e,r=eigsum(M,20)
            out.append(f'L{l2}H{j2}:{r:+.2f}/{np.abs(e[0]):.1f}')
    print(f'src L{l1}H{j1}:',' '.join(out))
```