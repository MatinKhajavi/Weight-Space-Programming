```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=250)
hd=64
def hm(l,name,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_{name}.{l}'][s]
Up,Sp,_=np.linalg.svd(W['W_proj.3']); L=Up[:,:27]
for l in range(4):
    lnl=W[f'ln1_w.{l}']
    row=[]
    for j in range(6):
        OV=W[f'W_o.{l}'][:,j*64:(j+1)*64]@(hm(l,'v',j)*lnl)
        row.append(f"H{j}:{np.linalg.norm(L.T@OV)**2/np.linalg.norm(OV)**2:.3f}")
    print(f"layer {l} OV output frac in logit subspace L (chance 0.070):"," ".join(row))
for l in range(4):
    Pj=W[f'W_proj.{l}']
    # per-neuron fraction in L, weighted
    fr=np.linalg.norm(L.T@Pj,axis=0)**2/np.linalg.norm(Pj,axis=0)**2
    print(f"MLP{l} output frac in L: overall {np.linalg.norm(L.T@Pj)**2/np.linalg.norm(Pj)**2:.3f}; neurons with >0.5 in L: {(fr>0.5).mean():.3f}")
```