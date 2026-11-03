```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=250)
hd=64
ln0=W['ln1_w.0']; ln2=W['ln2_w.0']
def hm(l,name,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_{name}.{l}'][s]
rng=np.random.default_rng(0)
Fc=W['W_fc.0']*ln2  # reads LN2'd residual
Fc_raw=W['W_fc.0']  # acts on normalized*ln2+... ; direction in raw space d -> ln2*d
def frac_along(d):
    d=d/np.linalg.norm(d)
    return np.linalg.norm(Fc@d)**2/np.linalg.norm(Fc)**2
print("chance:",1/384, "random dirs:", [round(frac_along(rng.standard_normal(384)),4) for _ in range(5)])
for j in [1,2,4,5,0,3]:
    Q=hm(0,'q',j)*ln0; K=hm(0,'k',j)*ln0
    U,S,Vt=np.linalg.svd(Q.T@K)
    for c in range(2):
        v=Vt[c]; u=U[:,c]
        # raw-space direction: v was folded with ln1_w (v_fold = v applies to normalized z: score uses (z*ln0)·K^T ... we folded ln into K so v is in z-space (normalized)). z-space is shared by LN1 and LN2 except weights; Fc (ln2 folded) also acts on z-space. So compare directly.
        print(f"L0H{j} comp{c}: k-dir frac in W_fc.0 = {frac_along(v):.4f}   q-dir frac = {frac_along(u):.4f}")
# Also V3 top input dirs and H5 OV input dirs for reference
V3=hm(0,'v',3)*ln0; _,_,Vt=np.linalg.svd(V3)
print("V3 in-dirs:", [round(frac_along(Vt[c]),4) for c in range(6)])
V5=hm(0,'v',5)*ln0; OV5=W['W_o.0'][:,320:384]@V5; _,_,Vt=np.linalg.svd(OV5)
print("OV5 in-dirs:", [round(frac_along(Vt[c]),4) for c in range(6)])
```