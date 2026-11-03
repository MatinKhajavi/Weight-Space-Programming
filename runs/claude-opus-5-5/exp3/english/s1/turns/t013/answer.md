```python
import numpy as np
for l in [1,2,3]:
    w=W[f'ln1_w.{l}']; b=W[f'ln1_b.{l}']
    for h in range(6):
        s=slice(64*h,64*h+64)
        Q=W[f'W_q.{l}'][s]*w; K=W[f'W_k.{l}'][s]*w
        qb=W[f'W_q.{l}'][s]@b
        u=K.T@qb
        i=np.argsort(-np.abs(u))[:8]
        frac=np.sum(u[i]**2)/np.sum(u**2)
        print(f'L{l}H{h} |u|={np.linalg.norm(u):.1f} top dims',i,np.round(u[i],2),f'frac top8={frac:.2f}')
```