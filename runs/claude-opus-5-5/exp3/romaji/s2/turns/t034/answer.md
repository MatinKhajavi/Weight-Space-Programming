```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
def headmats(l,h):
    lw=W[f'ln1_w.{l}']; s=slice(64*h,64*h+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
lw0=W['ln1_w.0']
F=W['W_fc.0']*W['ln2_w.0']; nf=np.linalg.norm(F,axis=1)
for thr,cap in [(0.05,40),(0.02,64),(0.1,30),(0.0,64)]:
    outs=[]
    for h in range(6):
        q,k,v,o=headmats(0,h); U,S,Vt=np.linalg.svd(o@v); r=int((S>S[0]*thr).sum()); outs.append(U[:,:min(r,cap)])
    Hout=np.linalg.svd(np.hstack(outs),full_matrices=False)[0]
    Hout=Hout[:, :min(Hout.shape[1],384)]
    for nthr in [1.0,2.0]:
        Fa=F[nf>nthr]
        Fr=Fa-(Fa@Hout)@Hout.T
        S=np.linalg.svd(Fr,compute_uv=False)
        print(thr,cap,'Hout dims',Hout.shape[1],'neurons',len(Fa),S[:34].round(2))
```