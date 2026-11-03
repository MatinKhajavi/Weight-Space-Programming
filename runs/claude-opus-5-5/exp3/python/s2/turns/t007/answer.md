```python
import numpy as np
w0=W['ln1_w.0']
for j in [0,1,2,3,5]:
    sl=slice(64*j,64*j+64)
    M=(W['W_q.0'][sl]*w0).T@(W['W_k.0'][sl]*w0)
    ev=np.linalg.eigvals(M)
    idx=np.argsort(-np.abs(ev))[:24]
    print(j, ' '.join(f'{abs(e):.2f}@{np.degrees(np.angle(e)):.0f}' for e in ev[idx]))
```