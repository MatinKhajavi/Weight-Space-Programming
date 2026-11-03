```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64
P=np.eye(d)-np.ones((d,d))/d
w0=W['ln1_w.0']
def blk(nm,l,j):
    s=slice(j*hd,(j+1)*hd); return W[f'{nm}.{l}'][s]
def rowspace(A,k):
    U,S,Vt=np.linalg.svd(A,full_matrices=False); return Vt[:k].T
def orth(A):
    Q,_=np.linalg.qr(A); return Q
T3=rowspace(blk('W_v',0,3),22)
Tres=orth(np.diag(1/w0)@T3)
OV3=W['W_o.0'][:,192:256]@blk('W_v',0,3)@np.diag(w0)   # acts on normalized residual
OV0=W['W_o.0'][:,0:64]@blk('W_v',0,0)@np.diag(w0)
def stats(C):
    ev=np.linalg.eigvals(C); return ev.real.sum()/np.abs(ev).sum(), np.sort(ev.real)[::-1][:4], np.sort(ev.real)[:3]
for l in range(1,4):
    wl=np.diag(W[f'ln1_w.{l}'])
    for j in range(6):
        QK=(blk('W_q',l,j)@wl).T@(blk('W_k',l,j)@wl)
        direct=Tres.T@QK@Tres
        ind3=Tres.T@QK@P@OV3@Tres
        ind0=Tres.T@QK@P@OV0@Tres
        s1=stats(direct); s2=stats(ind3); s3=stats(ind0)
        print(f"L{l}H{j}: direct T-T: asym={s1[0]:+.2f} top={s1[1]} bot={s1[2]} | ind via L0H3: asym={s2[0]:+.2f} top={s2[1]} bot={s2[2]} | via L0H0: asym={s3[0]:+.2f}")
```