```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
ln0=W['ln1_w.0']
def hm(l,name,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_{name}.{l}'][s]
V3=hm(0,'v',3)*ln0
_,_,Vt=np.linalg.svd(V3); T=Vt[:24].T
OV3=W['W_o.0'][:,192:256]@V3
Mlin=0.5*W['W_proj.0']@(W['W_fc.0']*W['ln2_w.0'])   # crude linearized MLP0
special=[21,39,58,99,184,383]
rng=np.random.default_rng(0)
def st(B):
    sym=(B+B.T)/2; ev=np.linalg.eigvalsh(sym)
    return f"{np.trace(B)/np.linalg.norm(B):+.2f}(mean eig {ev.mean():+.3f})"
print("tests: cur-vs-prev through MLP0-linearized; cur(MLP)-vs-prev(MLP); random-baseline std of tr/|B| ~1")
for l in [1,2,3]:
    lnl=W[f'ln1_w.{l}']
    for j in range(6):
        Q=(hm(l,'q',j)*lnl).copy(); K=(hm(l,'k',j)*lnl).copy(); Q[:,special]=0; K[:,special]=0
        QK=Q.T@K/8
        A1=T.T@QK@(Mlin@OV3)@T          # q=cur token direct, k=prev token via MLP0
        A2=T.T@(Mlin).T@QK@(Mlin@OV3)@T  # q=cur via MLP0, k=prev via MLP0
        A3=T.T@(Mlin).T@QK@(OV3)@T       # q=cur via MLP0, k=prev direct
        A4=T.T@(Mlin).T@QK@(Mlin)@T      # q=cur via MLP0, k=cur via MLP0 (same token)
        A5=T.T@(OV3).T@QK@(Mlin@OV3)@T   # q=prev direct, k=prev via MLP0
        print(f"L{l}H{j}: cur-prevM {st(A1)} | curM-prevM {st(A2)} | curM-prev {st(A3)} | curM-curM {st(A4)} | prev-prevM {st(A5)}")
```