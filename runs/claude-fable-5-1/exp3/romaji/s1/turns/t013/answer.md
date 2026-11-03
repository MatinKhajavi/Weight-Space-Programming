```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
ln0=W['ln1_w.0']
def hm(l,name,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_{name}.{l}'][s]
V3=hm(0,'v',3)*ln0
U,S,Vt=np.linalg.svd(V3)
T=Vt[:24].T   # 384 x 24 basis of token-ish subspace (as read by prev-token head)
OV3=W['W_o.0'][:,192:256]@V3
OV0=W['W_o.0'][:,0:64]@(hm(0,'v',0)*ln0)
special=[21,39,58,99,184,383]
rng=np.random.default_rng(0)
def stats(B):
    k=B.shape[0]
    sym=(B+B.T)/2
    ev=np.linalg.eigvalsh(sym)
    return f"tr/|B|={np.trace(B)/np.linalg.norm(B):+.2f} eig(sym) min={ev.min():+.3f} max={ev.max():+.3f} mean={ev.mean():+.3f}"
for l in [1,2,3]:
    lnl=W[f'ln1_w.{l}']
    print("=== layer",l)
    for j in range(6):
        Q=(hm(l,'q',j)*lnl).copy(); K=(hm(l,'k',j)*lnl).copy()
        Q[:,special]=0; K[:,special]=0
        QK=Q.T@K/8
        St=T.T@QK@T                 # same-token matching
        Bt=T.T@QK@OV3@T             # current token (q) vs previous token (k) via L0H3
        Bt2=T.T@OV3.T@QK@T          # previous token (q) vs current token (k)
        Bt3=T.T@OV3.T@QK@OV3@T      # prev vs prev
        print(f" H{j}: same-token: {stats(St)} | q=cur,k=prev: {stats(Bt)} | q=prev,k=cur: {stats(Bt2)} | prev-prev: {stats(Bt3)}")
# baseline: random 24-dim subspaces
print("baseline random subspace, layer1 H0:")
Q=(hm(1,'q',0)*W['ln1_w.1']).copy(); K=(hm(1,'k',0)*W['ln1_w.1']).copy(); Q[:,special]=0; K[:,special]=0; QK=Q.T@K/8
for _ in range(3):
    R,_=np.linalg.qr(rng.standard_normal((384,24)))
    print("  ",stats(R.T@QK@R), "|", stats(R.T@QK@OV3@R))
```