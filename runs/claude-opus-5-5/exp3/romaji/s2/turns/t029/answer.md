```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
def headmats(l,h):
    lw=W[f'ln1_w.{l}']; s=slice(64*h,64*h+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
lw0=W['ln1_w.0']
F=W['W_fc.0']*W['ln2_w.0']
nf=np.linalg.norm(F,axis=1)
print(np.histogram(nf,bins=20))
big=np.where(nf>1.0)[0]; print('n big',len(big))
# SVD of big neurons
Fb=F[big]
S=np.linalg.svd(Fb,compute_uv=False); print('sv big neurons',S[:40].round(2))
Fs=F[nf<0.8]; S2=np.linalg.svd(Fs,compute_uv=False); print('sv small neurons',S2[:40].round(2))
```