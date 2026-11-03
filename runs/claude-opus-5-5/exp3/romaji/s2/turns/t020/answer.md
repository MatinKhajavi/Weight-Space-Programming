```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
for l in [2,3]:
    lw=W[f'ln1_w.{l}']
    for h in range(6):
        s=slice(64*h,64*h+64)
        OV=W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*lw)
        S=np.linalg.svd(OV,compute_uv=False)
        print(l,h,S[:40].round(2))
```