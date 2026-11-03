```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def headmats(l,j):
    g=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
for j in range(6):
    Wq,Wk,Wv,Wo=headmats(3,j)
    s=np.linalg.svd(Wo@Wv,compute_uv=False)
    print(f'L3H{j} OV sv',s[:40])
    print('    Wo sv',np.linalg.svd(Wo,compute_uv=False)[:40])
```