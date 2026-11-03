```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64; H=6
def mats(l):
    gl=W[f"ln1_w.{l}"]; out=[]
    for j in range(H):
        s=slice(j*hd,(j+1)*hd)
        Wq,Wk,Wv,Wo=W[f"W_q.{l}"][s]*gl,W[f"W_k.{l}"][s]*gl,W[f"W_v.{l}"][s]*gl,W[f"W_o.{l}"][:,s]
        out.append((Wq.T@Wk, Wo@Wv))
    return out
allm=[mats(l) for l in range(4)]
def cos(A,B): return (A*B).sum()/np.linalg.norm(A)/np.linalg.norm(B)
for l in range(4):
    print(f"layer {l} QK cosine between heads:")
    print(np.array([[cos(allm[l][i][0],allm[l][j][0]) for j in range(H)] for i in range(H)]))
    print(f"layer {l} OV cosine between heads:")
    print(np.array([[cos(allm[l][i][1],allm[l][j][1]) for j in range(H)] for i in range(H)]))
# symmetric part analysis of layer-1 QK: top eigvecs of (QK+QK^T)/2 and the sign
for l in [1,2,3]:
    for j in range(H):
        QK=allm[l][j][0]; Ssym=(QK+QK.T)/2; ev=np.linalg.eigvalsh(Ssym)
        print(f"L{l}H{j} sym eigs top {ev[-4:][::-1]} bottom {ev[:4]}")
```