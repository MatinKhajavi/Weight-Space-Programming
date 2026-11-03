```python
import numpy as np
w0=W['ln1_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s[:k]
T,_=rs(np.vstack([W['W_v.0'][64:128]*w0, W['W_v.0'][320:384]*w0]),24)
Pq,_=rs(np.vstack([W['W_q.0'][64:128]*w0, W['W_q.0'][320:384]*w0, W['W_k.0'][64:128]*w0, W['W_k.0'][320:384]*w0]),30)
print('T vs P overlap', np.round(np.linalg.svd(T.T@Pq,compute_uv=False)[:10],2))
for j in [0,4]:
    sl=slice(64*j,64*j+64)
    for nm in ['W_q.0','W_k.0','W_v.0']:
        R,s=rs(W[nm][sl]*w0,4)
        print(f'H{j} {nm} frac in P30 {np.round(np.linalg.norm(Pq.T@R,axis=0)**2,2)}  in T {np.round(np.linalg.norm(T.T@R,axis=0)**2,2)}')
# where are H4 V top directions concentrated in coordinate basis?
R,s=rs(W['W_v.0'][256:320]*w0,3)
for i in range(3):
    v=R[:,i]; idx=np.argsort(-np.abs(v))[:10]
    print('H4 V dir',i,'top coords',idx, np.round(v[idx],2), 'ln1_w',np.round(w0[idx],2), 'PR',round(1/(v**4).sum(),1))
```