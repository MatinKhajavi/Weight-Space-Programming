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
print("H0 v (Q3 out):", sv(blk('W_v',0,0)@Pq)[:40])
print("H3 v (Q3 out):", sv(blk('W_v',0,3)@Pq)[:40])
st=np.vstack([blk('W_v',0,3),blk('W_v',0,0)])@Pq
print("H3+H0 v (Q3 out):", sv(st)[:45])
# OV of H3 and H0 in residual space, rank
w0=np.diag(W['ln1_w.0'])
for j in [0,3]:
    OV=W['W_o.0'][:,j*hd:(j+1)*hd]@blk('W_v',0,j)@w0@Pq
    print(f"H{j} OV (w0, Q3 out) sv:", sv(OV)[:40])
# The token channel as consumed by layer 1 (all V heads) and MLP1: from H0
P=np.eye(d)-np.ones((d,d))/d
OV0=W['W_o.0'][:,0:64]@blk('W_v',0,0)@w0
print("V1 P OV0 sv:", sv(W['W_v.1']@np.diag(W['ln1_w.1'])@P@OV0)[:30])
print("fc1 P OV0 sv:", sv(W['W_fc.1']@np.diag(W['ln2_w.1'])@P@OV0)[:30])
```