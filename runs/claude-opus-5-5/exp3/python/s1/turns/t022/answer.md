```python
import numpy as np
d=384; H=6; hd=64
def head(l,j):
    s=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    Q=W[f'W_q.{l}'][s]*g; K=W[f'W_k.{l}'][s]*g; V=W[f'W_v.{l}'][s]*g; O=W[f'W_o.{l}'][:,s]
    return Q,K,V,O
U3,s3,_=np.linalg.svd(W['W_proj.3'])
print(s3[:34].round(1))
for (l,j) in [(3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(2,1),(2,2)]:
    O=head(l,j)[3]
    e=np.linalg.norm(U3[:,:34].T@O,axis=1)/np.linalg.norm(O)*np.sqrt(384)
    print(f'L{l}H{j}', np.round(e,1))
```