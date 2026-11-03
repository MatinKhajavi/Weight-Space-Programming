```python
import numpy as np
d=384; H=6; hd=64
Uo0,so0,_=np.linalg.svd(W['W_o.0'])
Up0,sp0,_=np.linalg.svd(W['W_proj.0'])
for k_o,k_p in [(0,0),(2,0),(3,0),(3,1),(4,2),(6,3)]:
    B=np.hstack([Uo0[:,:k_o],Up0[:,:k_p]])
    if B.shape[1]>0:
        q,_=np.linalg.qr(B); P=np.eye(d)-q@q.T
    else: P=np.eye(d)
    s=np.linalg.svd(P@W['W_proj.3'],compute_uv=False)
    print(k_o,k_p, s[:34].round(1))
```