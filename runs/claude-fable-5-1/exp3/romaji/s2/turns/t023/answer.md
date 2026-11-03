```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64
P=np.eye(d)-np.ones((d,d))/d
w0=np.diag(W['ln1_w.0'])
def blk(nm,l,j):
    s=slice(j*hd,(j+1)*hd); return W[f'{nm}.{l}'][s]
def colspace(A,k):
    U,S,Vt=np.linalg.svd(A,full_matrices=False); return U[:,:k], S[:k]
def frac(A,B): return np.linalg.norm(A@B)**2/np.linalg.norm(A)**2
OVs={j:P@W['W_o.0'][:,j*hd:(j+1)*hd]@blk('W_v',0,j)@w0 for j in range(6)}
C5,S5=colspace(OVs[5],4); print("L0H5 out sv", S5)
C124,S124=colspace(np.hstack([OVs[1],OVs[2],OVs[4]]),6); print("L0H1/2/4 out sv", S124)
# orthogonalize C124 against C5 to separate
C124o,_=np.linalg.qr(C124-C5@(C5.T@C124))
print("principal cosines C5 vs C124:", np.linalg.svd(C5.T@C124,compute_uv=False))
for l in range(1,4):
    wl=np.diag(W[f'ln1_w.{l}'])
    for j in range(6):
        k=blk('W_k',l,j)@wl; q=blk('W_q',l,j)@wl
        print(f"L{l}H{j}: k in H5out(4)={frac(k,C5):.3f} [rand .010], k in H124out(6, orth to H5)={frac(k,C124o):.3f} [rand .016] | q: {frac(q,C5):.3f} {frac(q,C124o):.3f}")
```