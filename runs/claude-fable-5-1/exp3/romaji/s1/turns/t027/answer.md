```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
ln0=W['ln1_w.0']
def hm(l,name,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_{name}.{l}'][s]
V3=hm(0,'v',3)*ln0; _,_,VtV=np.linalg.svd(V3)
Fc=W['W_fc.0']*W['ln2_w.0']; _,_,VtF=np.linalg.svd(Fc)
Up,Sp,_=np.linalg.svd(W['W_proj.3'])
OV=W['W_o.3'][:,192:256]@(hm(3,'v',3)*W['ln1_w.3'])
for kL in [27,30,40,384]:
    L=Up[:,:kL]
    for name,Vt in [('V3',VtV),('Fc0',VtF)]:
        for kT in [28,32,40]:
            T=Vt[:kT].T
            S=np.linalg.svd(L.T@OV@T,compute_uv=False)
            print(f"L3H3 OV: L={kL} T={name}{kT}: svs {S[:32]}")
# Also L0H3 OV restricted to T input (how many token types the prev-token head copies)
OV3=W['W_o.0'][:,192:256]@V3
for name,Vt in [('V3',VtV),('Fc0',VtF)]:
    T=Vt[:32].T
    S=np.linalg.svd(OV3@T,compute_uv=False)
    print(f"L0H3 OV restricted to T={name}32: {S}")
```