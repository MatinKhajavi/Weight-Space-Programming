```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
ln0=W['ln1_w.0']
def hm(l,name,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_{name}.{l}'][s]
special=[21,39,58,99,184,383]
A=W['W_proj.3'].copy(); A[special,:]=0
Up,Sp,_=np.linalg.svd(A); L=Up[:,:24]
V3=hm(0,'v',3)*ln0; _,_,VtV=np.linalg.svd(V3)
Fc=W['W_fc.0']*W['ln2_w.0']; _,_,VtF=np.linalg.svd(Fc)
OV3=W['W_o.0'][:,192:256]@V3
Mlin=0.5*W['W_proj.0']@Fc
for name,Vt in [('V3',VtV),('Fc0',VtF)]:
    T=Vt[:28].T
    S=np.linalg.svd(L.T@OV3@T,compute_uv=False); print(f"direct prev->logit (T={name}): {S}")
    S=np.linalg.svd(L.T@Mlin@OV3@T,compute_uv=False); print(f"MLP0-linearized prev->logit (T={name}): {S}")
    S=np.linalg.svd(L.T@Mlin@T,compute_uv=False); print(f"MLP0-linearized cur->logit (T={name}): {S}")
    S=np.linalg.svd(L.T@T,compute_uv=False); print(f"direct cur->logit overlap (T={name}): {S}")
```