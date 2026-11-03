```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
for name in ['W_o.3','W_proj.3','W_o.2','W_proj.2','W_o.1','W_proj.1']:
    s=np.linalg.svd(W[name],compute_uv=False)
    print(name, s[:45])
M=np.concatenate([W['W_o.3'],W['W_proj.3']],1)
s=np.linalg.svd(M,compute_uv=False); print('concat L3', s[:45])
```