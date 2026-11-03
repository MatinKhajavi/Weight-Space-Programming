```python
import numpy as np
w0=W['ln1_w.0']
for j in [1,5]:
    sl=slice(64*j,64*j+64)
    OV=W['W_o.0'][:,sl]@(W['W_v.0'][sl]*w0)
    s=np.linalg.svd(OV,compute_uv=False)
    print(j, np.round(s[:45],3))
    O=W['W_o.0'][:,sl]; print(' O sv',np.round(np.linalg.svd(O,compute_uv=False)[:64],2))
```