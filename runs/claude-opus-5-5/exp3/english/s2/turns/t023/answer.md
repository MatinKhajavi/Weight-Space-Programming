```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def headmats(l,j):
    g=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
Up,sp,_=np.linalg.svd(W['W_proj.3'],full_matrices=False)
outs={}
for j in range(6):
    Wq,Wk,Wv,Wo=headmats(3,j); U,s,_=np.linalg.svd(Wo@Wv,full_matrices=False); outs[j]=(U,s)
print('cos of principal angles MLP3 top-40 vs L3H1 top-40:',np.linalg.svd(Up[:,:40].T@outs[1][0][:,:40],compute_uv=False))
for k in [23,24,27,28,29,30]:
    S=Up[:,:k]
    print(k, 'energy frac of each L3 head OV inside MLP3 top-k:', [round(float(np.linalg.norm(S.T@(outs[j][0]*outs[j][1]))**2/np.sum(outs[j][1]**2)),3) for j in range(6)])
# also how much of each L3 head's OV energy lies in the subspace vs random baseline k/384
```