```python
import numpy as np
for l in [2,3]:
    Pj=W[f'W_proj.{l}']; Wo=W[f'W_o.{l}']
    s1=np.linalg.svd(Pj,compute_uv=False); s2=np.linalg.svd(Wo,compute_uv=False)
    s3=np.linalg.svd(np.hstack([Pj,Wo]),compute_uv=False)
    print(l,'W_proj sv',np.round(s1[:40],1)); print('  W_o sv',np.round(s2[:40],1)); print('  both',np.round(s3[:40],1))
```