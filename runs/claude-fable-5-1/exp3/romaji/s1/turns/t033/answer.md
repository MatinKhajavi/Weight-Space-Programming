```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=250)
hd=64
ln0=W['ln1_w.0']
def hm(l,name,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_{name}.{l}'][s]
V3=hm(0,'v',3)*ln0; _,_,VtV=np.linalg.svd(V3); T=VtV[:24].T
OV3=W['W_o.0'][:,192:256]@V3; Uo,_,_=np.linalg.svd(OV3); H3out=Uo[:,:20]
OV5=W['W_o.0'][:,320:384]@(hm(0,'v',5)*ln0); Uo5,_,_=np.linalg.svd(OV5); H5rich=Uo5[:,2:8]
Q3=hm(0,'q',3)*ln0; K3=hm(0,'k',3)*ln0; _,_,VtP=np.linalg.svd(np.vstack([Q3,K3])); P=VtP[:24].T
Fc=W['W_fc.0']*W['ln2_w.0']; Pj=W['W_proj.0']
for d in [21,184,39,58,99,383]:
    w=Pj[d]  # 1536 weights into dim d
    idx=np.argsort(-np.abs(w))[:12]
    print(f"dim {d}: total |row|={np.linalg.norm(w):.2f}; top neurons:")
    for n in idx:
        f=Fc[n]; tot=np.linalg.norm(f)**2
        print(f"   n{n}: w={w[n]:+.2f} out|o_n|={np.linalg.norm(Pj[:,n]):.2f} frac_to_dim={w[n]**2/np.linalg.norm(Pj[:,n])**2:.2f} | in: T={np.linalg.norm(f@T)**2/tot:.2f} prev={np.linalg.norm(f@H3out)**2/tot:.2f} H5rich={np.linalg.norm(f@H5rich)**2/tot:.2f} P={np.linalg.norm(f@P)**2/tot:.2f} |f|={np.sqrt(tot):.2f}")
```