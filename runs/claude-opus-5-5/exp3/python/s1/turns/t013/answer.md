```python
import numpy as np
d=384; H=6; hd=64
def head(l,j):
    s=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    Q=W[f'W_q.{l}'][s]*g; K=W[f'W_k.{l}'][s]*g; V=W[f'W_v.{l}'][s]*g; O=W[f'W_o.{l}'][:,s]
    return Q,K,V,O
for j in range(H):
    Q,K,V,O=head(0,j)
    print(j,'V ',np.linalg.svd(V,compute_uv=False)[:40].round(3))
    print(j,'OV',np.linalg.svd(O@V,compute_uv=False)[:40].round(3))
```