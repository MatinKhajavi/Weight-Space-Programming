```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64
P=np.eye(d)-np.ones((d,d))/d
w0=W['ln1_w.0']
def blk(nm,l,j):
    s=slice(j*hd,(j+1)*hd); return W[f'{nm}.{l}'][s]
def sv(a): return np.linalg.svd(a,compute_uv=False)
def rowspace(A,k):
    U,S,Vt=np.linalg.svd(A,full_matrices=False); return Vt[:k].T
Q3=rowspace(np.vstack([blk('W_q',0,3),blk('W_k',0,3)]),30)
Pq=np.eye(d)-Q3@Q3.T
fc=W['W_fc.0']; A=fc[np.linalg.norm(fc,axis=1)>1]
print("fc.0 trained rows, Q3 projected out, sv:", sv(A@Pq)[:50])
print("W_v.0[H3] Q3 projected out, sv:", sv(blk('W_v',0,3)@Pq)[:40])
# downstream usage of L0H3 output by layer-1 values and MLP1
OV3=W['W_o.0'][:,192:256]@blk('W_v',0,3)@np.diag(w0)
V1=W['W_v.1']@np.diag(W['ln1_w.1'])
print("V1 P OV3 sv:", sv(V1@P@OV3)[:40])
F1=W['W_fc.1']@np.diag(W['ln2_w.1'])
print("fc1 P OV3 sv:", sv(F1@P@OV3)[:40])
# and of MLP0 output by layer-1 values
print("V1 P Wproj0 sv:", sv(V1@P@W['W_proj.0'])[:40])
print("Wproj0 sv:", sv(W['W_proj.0'])[:40])
# OV3 itself (with w0) spectrum
print("OV3 (with w0) sv:", sv(OV3)[:40])
```