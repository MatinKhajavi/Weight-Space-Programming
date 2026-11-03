```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
g=W['ln1_w.0']
for j in range(6):
    s=slice(64*j,64*j+64)
    print(j,'Wv raw sv',np.linalg.svd(W['W_v.0'][s],compute_uv=False)[:40])
    print(j,'Wv*g sv',np.linalg.svd(W['W_v.0'][s]*g,compute_uv=False)[:40])
```