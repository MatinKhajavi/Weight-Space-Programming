```python
import numpy as np
A=np.vstack([W['W_q.0'],W['W_k.0'],W['W_v.0']])
s=np.linalg.svd(A,compute_uv=False)
print(s[:220].round(2))
print(s[220:].round(3))
```