```python
import numpy as np
A=np.vstack([W['W_q.0'],W['W_k.0'],W['W_v.0'],W['W_fc.0']])
s=np.linalg.svd(A,compute_uv=False)
print(s[:200].round(2))
B=np.hstack([W['W_o.0'],W['W_proj.0']])
s=np.linalg.svd(B,compute_uv=False)
print(s[:200].round(2))
```