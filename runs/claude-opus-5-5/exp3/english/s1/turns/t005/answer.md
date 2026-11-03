```python
import numpy as np
def eff(l):
    w=W[f'ln1_w.{l}']; b=W[f'ln1_b.{l}']
    return W[f'W_q.{l}']*w, W[f'W_k.{l}']*w, W[f'W_v.{l}']*w, W[f'W_o.{l}']
for l,h in [(0,2),(0,3),(1,2),(2,0),(3,1)]:
    Q,K,V,O=eff(l)
    s=slice(64*h,64*h+64)
    B=Q[s].T@K[s]
    ev=np.linalg.eigvals(B); ev=ev[np.argsort(-np.abs(ev))][:40]
    print(f'L{l}H{h}')
    print('  |ev|', np.round(np.abs(ev),2))
    print('  angle/pi', np.round(np.angle(ev)/np.pi,3))
```