```python
import numpy as np
w=W['ln1_w.0']; print(np.round(np.sort(w)[::-1][:40],3)); print(np.round(np.sort(w)[:40],3))
print(np.histogram(w,bins=20))
for k in ['W_q.0','W_k.0','W_v.0','W_fc.0']:
    s=np.linalg.svd(W[k],compute_uv=False); print(k, np.round(s[:60],2), np.round(s[[80,100,150,200,300,383]],3))
```