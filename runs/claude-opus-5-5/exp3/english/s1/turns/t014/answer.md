```python
import numpy as np
w=W['ln1_w.0']
s2=slice(128,192)
Q2=W['W_q.0'][s2]*w; K2=W['W_k.0'][s2]*w; V2=W['W_v.0'][s2]*w
def topsub(M,k):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return Vt[:k].T
Pq=topsub(Q2,20); Pk=topsub(K2,20); Pv=topsub(V2,18)
def overlap(A,B):  # mean squared cosines of principal angles
    s=np.linalg.svd(A.T@B,compute_uv=False); return np.round(s,2)
print('Q2 vs K2 subspace', overlap(Pq,Pk))
print('Q2 vs V2', overlap(Pq,Pv))
for h in [0,1,4,5,3]:
    s=slice(64*h,64*h+64)
    OV=W['W_o.0'][:,s]@(W['W_v.0'][s]*w)
    for d in [222,78,72]:
        a=OV[d]; a2=a/np.linalg.norm(a)
        print(f'L0H{h} row{d} |a|={np.linalg.norm(a):.2f} frac in Q2sub={np.sum((Pq.T@a2)**2):.2f} K2sub={np.sum((Pk.T@a2)**2):.2f} V2sub={np.sum((Pv.T@a2)**2):.2f}')
```