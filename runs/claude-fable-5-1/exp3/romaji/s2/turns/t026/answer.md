```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64
def blk(nm,l,j):
    s=slice(j*hd,(j+1)*hd); return W[f'{nm}.{l}'][s]
def rowspace(A,k):
    U,S,Vt=np.linalg.svd(A,full_matrices=False); return Vt[:k].T
def sv(a): return np.linalg.svd(a,compute_uv=False)
qk3=np.vstack([blk('W_q',0,3),blk('W_k',0,3)])
print("L0H3 q+k stacked sv (position subspace size):", sv(qk3)[:70])
for k in [30,45,60]:
    Q=rowspace(qk3,k); Pq=np.eye(d)-Q@Q.T
    print(f"Q3({k}) out: H3 v sv:", sv(blk('W_v',0,3)@Pq)[:30])
    print(f"Q3({k}) out: H3+H0 v sv:", sv(np.vstack([blk('W_v',0,3),blk('W_v',0,0)])@Pq)[:34])
    print(f"Q3({k}) out: all v sv:", sv(W['W_v.0']@Pq)[:36])
```