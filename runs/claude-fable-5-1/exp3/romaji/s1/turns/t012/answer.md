```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=250)
hd=64
ln0=W['ln1_w.0']
def hm(l,name,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_{name}.{l}'][s]
V3=hm(0,'v',3)*ln0
U,S,Vt=np.linalg.svd(V3)
print("sv of V3:",S[:30])
# token subspace candidate from V3's top right singular vectors
# also from W_fc.0 (ln2 folded) top right singular vectors
Fc=W['W_fc.0']*W['ln2_w.0']
Uf,Sf,Vtf=np.linalg.svd(Fc)
print("sv of W_fc.0 (ln2-folded):",Sf[:40])
# overlap between V3 subspace and Fc subspace
for k in [16,24,32]:
    A=Vt[:k]; B=Vtf[:k]
    print(f"k={k}: overlap(V3 subspace, Fc subspace) = {np.linalg.norm(A@B.T)**2/k:.3f} (chance {k/384:.3f})")
# Fractions of later-layer Q,K,V weights in T (V3 top-24), excluding special dims
T=Vt[:24]
special=[21,39,58,99,184,383]
for l in [1,2,3]:
    lnl=W[f'ln1_w.{l}']
    res=[]
    for name in ['q','k','v']:
        for j in range(6):
            A=(hm(l,name,j)*lnl).copy(); A[:,special]=0
            fr=np.linalg.norm(A@T.T)**2/np.linalg.norm(A)**2
            res.append(f"{name}{j}:{fr:.2f}")
    print(f"L{l} frac in T(V3 top24) chance {24/384:.3f}:"," ".join(res))
T=Vtf[:24]
for l in [1,2,3]:
    lnl=W[f'ln1_w.{l}']
    res=[]
    for name in ['q','k','v']:
        for j in range(6):
            A=(hm(l,name,j)*lnl).copy(); A[:,special]=0
            fr=np.linalg.norm(A@T.T)**2/np.linalg.norm(A)**2
            res.append(f"{name}{j}:{fr:.2f}")
    print(f"L{l} frac in T(Fc top24):"," ".join(res))
# Later layers' MLPs reading T
for l in [1,2,3]:
    F=W[f'W_fc.{l}']*W[f'ln2_w.{l}']
    print(f"W_fc.{l} frac in T(V3 top24): {np.linalg.norm(F@Vt[:24].T)**2/np.linalg.norm(F)**2:.3f}, in T(Fc0 top24): {np.linalg.norm(F@Vtf[:24].T)**2/np.linalg.norm(F)**2:.3f}, in special dims: {np.linalg.norm(F[:,special])**2/np.linalg.norm(F)**2:.3f}")
```