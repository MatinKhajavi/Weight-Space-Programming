```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64
def blk(nm,l,j):
    s=slice(j*hd,(j+1)*hd); return W[f'{nm}.{l}'][s]
def rowspace(A,k):
    U,S,Vt=np.linalg.svd(A,full_matrices=False); return Vt[:k].T
def sv(a): return np.linalg.svd(a,compute_uv=False)
Q3=rowspace(np.vstack([blk('W_q',0,3),blk('W_k',0,3)]),40)
Pq=np.eye(d)-Q3@Q3.T
allv=np.vstack([blk('W_v',0,j) for j in range(6)])@Pq
print("all W_v.0 (Q3 out):", sv(allv)[:45])
allk=np.vstack([blk('W_k',0,j) for j in [0,1,2,4,5]])@Pq
print("W_k.0 H0,1,2,4,5 (Q3 out):", sv(allk)[:45])
allq=np.vstack([blk('W_q',0,j) for j in [0,1,2,4,5]])@Pq
print("W_q.0 H0,1,2,4,5 (Q3 out):", sv(allq)[:45])
# subspace agreement: top-24 of allv vs top-24 of H3 v, principal cosines
A=rowspace(allv,24); B=rowspace(blk('W_v',0,3)@Pq,20)
print("principal cosines allv(24) vs H3v(20):", np.linalg.svd(A.T@B,compute_uv=False))
fc=W['W_fc.0']; F=fc[np.linalg.norm(fc,axis=1)>1]@Pq
C=rowspace(F,30)
print("principal cosines fc(30) vs H3v(20):", np.linalg.svd(C.T@B,compute_uv=False))
print("principal cosines fc(30) vs allv(24):", np.linalg.svd(C.T@A,compute_uv=False))
```