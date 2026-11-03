```python
import numpy as np
A=np.vstack([W['W_q.0'],W['W_k.0'],W['W_v.0']])
s=np.linalg.svd(A,compute_uv=False)
print(np.round(s[:200],3))
print(np.round(s[200:],3))
w=W['ln1_w.0']
s2=np.linalg.svd(A*w,compute_uv=False)
print('with lnw', np.round(s2[:200],3))
```