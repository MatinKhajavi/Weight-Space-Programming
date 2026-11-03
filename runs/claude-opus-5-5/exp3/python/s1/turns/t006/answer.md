```python
import numpy as np
d=384; H=6; hd=64
def head(l,j):
    s=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    Q=W[f'W_q.{l}'][s]*g; K=W[f'W_k.{l}'][s]*g; V=W[f'W_v.{l}'][s]*g; O=W[f'W_o.{l}'][:,s]
    return Q,K,V,O
for l in range(4):
    for j in range(H):
        Q,K,V,O=head(l,j)
        sov=np.linalg.svd(O@V,compute_uv=False)
        sqk=np.linalg.svd(Q.T@K,compute_uv=False)
        print(l,j,'OV',sov[:6].round(2),'QK',sqk[:6].round(2), 'normQ',np.linalg.norm(Q).round(2),'normK',np.linalg.norm(K).round(2))
```