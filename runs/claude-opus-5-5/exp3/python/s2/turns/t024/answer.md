```python
import numpy as np
w0=W['ln1_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s[:k]
T,_=rs(np.vstack([W['W_v.0'][64:128]*w0, W['W_v.0'][320:384]*w0]),32)
for kP in [20,30,40]:
    P,_=rs(np.vstack([W['W_q.0'][64:128]*w0, W['W_q.0'][320:384]*w0, W['W_k.0'][64:128]*w0, W['W_k.0'][320:384]*w0]),kP)
    print('kP',kP,'T vs P overlap top', np.round(np.linalg.svd(T.T@P,compute_uv=False)[:6],2))
    for j in [0,2,3,4]:
        sl=slice(64*j,64*j+64)
        for nm in ['W_q.0','W_k.0','W_v.0']:
            R,s=rs(W[nm][sl]*w0,4)
            print(f'  H{j} {nm} sv{np.round(s,2)} inP {np.round(np.linalg.norm(P.T@R,axis=0)**2,2)}  inT {np.round(np.linalg.norm(T.T@R,axis=0)**2,2)}')
```