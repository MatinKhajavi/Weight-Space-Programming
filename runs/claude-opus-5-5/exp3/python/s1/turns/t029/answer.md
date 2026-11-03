```python
import numpy as np
d=384; H=6; hd=64
def head(l,j):
    s=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    Q=W[f'W_q.{l}'][s]*g; K=W[f'W_k.{l}'][s]*g; V=W[f'W_v.{l}'][s]*g; O=W[f'W_o.{l}'][:,s]
    return Q,K,V,O
Vt=np.vstack([head(0,j)[2]/np.linalg.norm(head(0,j)[2]) for j in [1,5]])
_,sv,vt=np.linalg.svd(Vt,full_matrices=False)
Ptok=vt[:26].T
U3,s3,_=np.linalg.svd(W['W_proj.3'])
for (l,j) in [(3,1),(3,5),(2,1),(3,0)]:
    Q,K,V,O=head(l,j)
    OVt=O@V@Ptok
    OVall=O@V
    a=np.linalg.norm(U3[:,:34].T@OVt,axis=1)
    b=np.linalg.norm(U3[:,:34].T@OVall,axis=1)
    print(f'L{l}H{j} tok->dir', np.round(a,2))
    print(f'      frac  ', np.round(a/b,2))
# random baseline for frac: sqrt(26/384)
print(np.sqrt(26/384))
```