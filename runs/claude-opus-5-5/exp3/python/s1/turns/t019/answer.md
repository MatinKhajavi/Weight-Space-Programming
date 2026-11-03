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
OVs['I']=np.eye(d)
for l in [1,2,3]:
    for j in range(H):
        Q,K,V,O=head(l,j)
        QK=Q.T@K
        res=[]
        for qh,kh in [('I',5),('I',1),('I',2),('I',3),(5,1),(1,5),(5,2),(2,5),(5,3),(3,5),(1,3),(3,1),(1,2),(2,1)]:
            r,m=pos_ratio(OVs[qh].T@QK@OVs[kh])
            res.append(f'q{qh}k{kh}:{r:+.2f}')
        print(f'L{l}H{j}',' '.join(res))
```