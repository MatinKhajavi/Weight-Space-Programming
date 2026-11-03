```python
import numpy as np
d=384; H=6; hd=64
def head(l,j):
    s=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    Q=W[f'W_q.{l}'][s]*g; K=W[f'W_k.{l}'][s]*g; V=W[f'W_v.{l}'][s]*g; O=W[f'W_o.{l}'][:,s]
    return Q,K,V,O
def pos_ratio(M,k=20):
    e=np.linalg.eigvals(M); idx=np.argsort(-np.abs(e))[:k]; e=e[idx]
    return e.real.sum()/np.abs(e).sum(), np.abs(e[0])
OVs={h:head(0,h)[3]@head(0,h)[2] for h in range(6)}
for l in [1,2,3]:
    for j in range(H):
        Q,K,V,O=head(l,j)
        QK=Q.T@K
        res=[]
        for h in [1,5,2,3]:
            A=pos_ratio(QK@OVs[h]); B=pos_ratio(OVs[h].T@QK); C=pos_ratio(OVs[h].T@QK@OVs[h])
            res.append(f'h{h} A{A[0]:+.2f}({A[1]:.2f}) B{B[0]:+.2f}({B[1]:.2f}) C{C[0]:+.2f}({C[1]:.2f})')
        print(f'L{l}H{j}', ' | '.join(res))
```