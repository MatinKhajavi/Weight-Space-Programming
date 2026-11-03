```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
def hm(l,name,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_{name}.{l}'][s]
Up,Sp,_=np.linalg.svd(W['W_proj.3']); L=Up[:,:27]
for l in [3,2]:
    for j in range(6):
        OV=W[f'W_o.{l}'][:,j*64:(j+1)*64]@(hm(l,'v',j)*W[f'ln1_w.{l}'])
        fr=np.linalg.norm(L.T@OV)**2/np.linalg.norm(OV)**2
        S=np.linalg.svd(L.T@OV,compute_uv=False)
        print(f"L{l}H{j}: frac of OV output in L = {fr:.3f}; svs of L^T OV: {S[:27]}")
# MLP2 output in L
for l in [2,1,0]:
    Pj=W[f'W_proj.{l}']; print(f"W_proj.{l} frac in L: {np.linalg.norm(L.T@Pj)**2/np.linalg.norm(Pj)**2:.3f}")
```