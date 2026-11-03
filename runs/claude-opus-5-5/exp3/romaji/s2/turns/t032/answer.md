```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
def headmats(l,h):
    lw=W[f'ln1_w.{l}']; s=slice(64*h,64*h+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
lw0=W['ln1_w.0']
# all L0 head output subspaces (top 30 each)
outs=[]
for h in range(6):
    q,k,v,o=headmats(0,h); U,S,Vt=np.linalg.svd(o@v); r=int((S>S[0]*0.05).sum()); outs.append(U[:,:min(r,40)]); print(h,'rank used',min(r,40))
Hout=np.linalg.svd(np.hstack(outs),full_matrices=False)[0]
print('head-out basis dims',Hout.shape)
F=W['W_fc.0']*W['ln2_w.0']; nf=np.linalg.norm(F,axis=1); Fa=F[nf>1.0]
Fr=Fa-(Fa@Hout)@Hout.T
S=np.linalg.svd(Fr,compute_uv=False); print('residual (tok+pos) sv',S[:50].round(2))
# also attention reading of L0 Q,K,V (in ln-scaled -> raw): stack
```