```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=250)
def headmats(l,h):
    lw=W[f'ln1_w.{l}']; s=slice(64*h,64*h+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
F=W['W_fc.0']*W['ln2_w.0']; nf=np.linalg.norm(F,axis=1)
outs=[]
for h in range(6):
    q,k,v,o=headmats(0,h); U,S,Vt=np.linalg.svd(o@v); r=int((S>S[0]*0.05).sum()); outs.append(U[:,:min(r,40)])
Hout=np.linalg.svd(np.hstack(outs),full_matrices=False)[0]
Fa=F[nf>1.0]; Fr=Fa-(Fa@Hout)@Hout.T
U,S,Vt=np.linalg.svd(Fr,full_matrices=False); D=Vt[:26].T
for l in [1,2,3]:
    for h in range(6):
        q,k,v,o=headmats(l,h); M=D.T@(q.T@k/8)@D
        Ms=(M+M.T)/2; ev=np.linalg.eigvalsh(Ms)
        sv=np.linalg.svd(M,compute_uv=False)
        print(l,h,'tr %.3f'%np.trace(M),'sym eig lowest',ev[:4].round(3),'highest',ev[-4:].round(3),'sv',sv[:5].round(3))
```