```python
import numpy as np
def eff(l):
    w=W[f'ln1_w.{l}']; b=W[f'ln1_b.{l}']
    return W[f'W_q.{l}']*w, W[f'W_k.{l}']*w, W[f'W_v.{l}']*w, W[f'W_o.{l}'], W[f'W_q.{l}']@b, W[f'W_k.{l}']@b, W[f'W_v.{l}']@b
for l in range(4):
    Q,K,V,O,qb,kb,vb=eff(l)
    for h in range(6):
        s=slice(64*h,64*h+64)
        u=K[s].T@qb[s]  # key-side preference from query bias
        r=Q[s].T@kb[s]  # query-side term (constant over keys -> irrelevant for softmax)
        print(f'L{l}H{h} |qbias|={np.linalg.norm(qb[s]):.2f} |kbias|={np.linalg.norm(kb[s]):.2f} |u|={np.linalg.norm(u):.3f} score range ~ {np.linalg.norm(u)*np.sqrt(384)/8:.2f}   |B|*384/8={np.linalg.norm(Q[s].T@K[s],2)*384/8:.1f}')
```