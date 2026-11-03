```python
import numpy as np
w0=W['ln1_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s[:k]
T,_=rs(np.vstack([W['W_v.0'][64:128]*w0, W['W_v.0'][320:384]*w0]),24)
for j in [0,4,2,3]:
    sl=slice(64*j,64*j+64)
    for nm in ['W_q.0','W_k.0','W_v.0']:
        R,s=rs(W[nm][sl]*w0,4)
        fr=np.linalg.norm(T.T@R,axis=0)**2
        print(f'H{j} {nm} sv {np.round(s,2)} frac in T {np.round(fr,2)}')
```