```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
ln0=W['ln1_w.0']
def hm(l,name,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_{name}.{l}'][s]
V3=hm(0,'v',3)*ln0; OV3=W['W_o.0'][:,192:256]@V3
_,_,VtV=np.linalg.svd(V3)
Fc=W['W_fc.0']*W['ln2_w.0']; _,_,VtF=np.linalg.svd(Fc)
V0=hm(0,'v',0)*ln0; _,_,Vt0=np.linalg.svd(V0)
special=[21,39,58,99,184,383]
rng=np.random.default_rng(0)
def stat(B):
    sym=(B+B.T)/2; ev=np.linalg.eigvalsh(sym)
    return np.trace(B)/np.linalg.norm(B), ev.mean()/np.abs(ev).mean()
for (l,j) in [(3,3),(3,0),(3,5),(2,4),(1,1)]:
    lnl=W[f'ln1_w.{l}']
    Q=(hm(l,'q',j)*lnl).copy(); K=(hm(l,'k',j)*lnl).copy(); Q[:,special]=0; K[:,special]=0
    QK=Q.T@K/8
    print(f"L{l}H{j}:")
    for name,Vt in [('V3',VtV),('Fc0',VtF),('V0',Vt0)]:
        for k in [12,16,24,32]:
            T=Vt[:k].T
            t,e=stat(T.T@QK@OV3@T)
            # baseline: random k-dim subspaces
            bs=[stat(R.T@QK@OV3@R)[0] for R in [np.linalg.qr(rng.standard_normal((384,k)))[0] for _ in range(20)]]
            print(f"   T={name} k={k}: tr/|B|={t:+.2f} (sym-eig-mean ratio {e:+.2f})  baseline {np.mean(bs):+.2f}±{np.std(bs):.2f}")
```