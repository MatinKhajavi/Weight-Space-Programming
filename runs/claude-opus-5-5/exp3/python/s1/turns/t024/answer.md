```python
import numpy as np
d=384; H=6; hd=64
def head(l,j):
    s=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    Q=W[f'W_q.{l}'][s]*g; K=W[f'W_k.{l}'][s]*g; V=W[f'W_v.{l}'][s]*g; O=W[f'W_o.{l}'][:,s]
    return Q,K,V,O
Vs=np.vstack([head(0,j)[2]/np.linalg.norm(head(0,j)[2]) for j in [1,5,2]])
_,sv,vt=np.linalg.svd(Vs,full_matrices=False)
Ptok=vt[:28].T
prefs={}
for l in [1,2,3]:
    for j in range(H):
        s=slice(j*hd,(j+1)*hd)
        Q,K,V,O=head(l,j)
        q0=W[f'W_q.{l}'][s]@W[f'ln1_b.{l}']
        b=(q0@K@Ptok)/8
        Mk=Q.T@K@Ptok/8
        sk=np.linalg.svd(Mk,compute_uv=False)
        prefs[(l,j)]=b
        print(f'L{l}H{j} |bias key-pref|={np.linalg.norm(b):.2f}  QK(any->tok key) sv {sk[:6].round(2)}')
keys=list(prefs)
B=np.array([prefs[k]/np.linalg.norm(prefs[k]) for k in keys])
print(np.round(B@B.T,2))
```