```python
import numpy as np
w0=W['ln1_w.0']
def QK(l,j):
    w=W[f'ln1_w.{l}']; sl=slice(64*j,64*j+64)
    return (W[f'W_q.{l}'][sl]*w).T@(W[f'W_k.{l}'][sl]*w)
D=np.load('/tmp/D.npy')
for j in [0,2,3]:
    M=QK(0,j)
    MD=M@D.T     # 384 x 28 : key-side contributions
    u,s,vt=np.linalg.svd(MD)
    q=u[:,0]; sc=q@MD
    if np.abs(sc.max())<np.abs(sc.min()): sc=-sc
    o=np.argsort(-sc)
    print(f'H{j} key token scores (dominant query dir, sv {s[:3].round(2)}):',' '.join(f'{c}:{sc[c]:+.2f}' for c in o))
```