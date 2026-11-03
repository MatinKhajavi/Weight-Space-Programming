```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
ln0=W['ln1_w.0']
def hm(l,name,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_{name}.{l}'][s]
def orth(A):  # orthonormal basis of column space of A (384 x k)
    q,_=np.linalg.qr(A); return q
V3=hm(0,'v',3)*ln0
U,S,Vt=np.linalg.svd(V3); T=Vt[:24].T
OV3=W['W_o.0'][:,192:256]@V3
Uo,So,Vto=np.linalg.svd(OV3); H3out=Uo[:,:24]
OV0=W['W_o.0'][:,0:64]@(hm(0,'v',0)*ln0); Uo,So,_=np.linalg.svd(OV0); H0out=Uo[:,:24]
OV5=W['W_o.0'][:,320:384]@(hm(0,'v',5)*ln0); Uo,So,_=np.linalg.svd(OV5); H5out=Uo[:,2:26]  # exclude the 2 special dirs
Up,Sp,_=np.linalg.svd(W['W_proj.0']); M0out=Up[:,:48]
Up,Sp,_=np.linalg.svd(W['W_proj.1']); M1out=Up[:,:48]
Up,Sp,_=np.linalg.svd(W['W_proj.2']); M2out=Up[:,:48]
subs={'T(tok)':T,'H3out(prev)':H3out,'H0out':H0out,'H5out(rich)':H5out,'MLP0out48':M0out,'MLP1out48':M1out,'MLP2out48':M2out}
special=[21,39,58,99,184,383]
for l in [1,2,3]:
    lnl=W[f'ln1_w.{l}']
    print(f"=== layer {l} (chance: 24-dim {24/384:.3f}, 48-dim {48/384:.3f})")
    for j in range(6):
        line=f" H{j}:"
        for name in ['q','k','v']:
            A=(hm(l,name,j)*lnl).copy(); A[:,special]=0
            tot=np.linalg.norm(A)**2
            line+=f" {name}["+" ".join(f"{sn}={np.linalg.norm(A@B)**2/tot:.2f}" for sn,B in subs.items())+"]"
        print(line)
    F=W[f'W_fc.{l}']*W[f'ln2_w.{l}']; tot=np.linalg.norm(F)**2
    print(f" MLP{l} in: "+" ".join(f"{sn}={np.linalg.norm(F@B)**2/tot:.2f}" for sn,B in subs.items()))
```