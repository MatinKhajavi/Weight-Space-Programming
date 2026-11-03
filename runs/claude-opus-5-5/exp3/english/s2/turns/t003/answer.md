```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
g=W['ln1_w.0']
print(np.sort(g)[:40]); print(np.sort(g)[-20:])
print(np.histogram(g,bins=20))
g2=W['ln2_w.0']; print(np.histogram(g2,bins=20))
print(np.corrcoef(g,g2))
for j in [2,3]:
    s=slice(64*j,64*j+64)
    A=(W['W_q.0'][s]*g).T@(W['W_k.0'][s]*g)
    ev=np.linalg.eigvals(A)
    idx=np.argsort(-np.abs(ev))[:40]
    print('head',j)
    for e in ev[idx]: print('%.2f ang %.1f deg'%(abs(e),np.degrees(np.angle(e))), end='; ')
    print()
```