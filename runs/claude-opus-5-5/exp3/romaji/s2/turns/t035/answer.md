```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
def headmats(l,h):
    lw=W[f'ln1_w.{l}']; s=slice(64*h,64*h+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
lw0=W['ln1_w.0']
F=W['W_fc.0']*W['ln2_w.0']; nf=np.linalg.norm(F,axis=1)
outs=[]
for h in range(6):
    q,k,v,o=headmats(0,h); U,S,Vt=np.linalg.svd(o@v); r=int((S>S[0]*0.05).sum()); outs.append(U[:,:min(r,40)])
Hout=np.linalg.svd(np.hstack(outs),full_matrices=False)[0]
Fa=F[nf>1.0]; Fr=Fa-(Fa@Hout)@Hout.T
U,S,Vt=np.linalg.svd(Fr,full_matrices=False); D=Vt[:34].T
# raw-coordinate readers: Wv*lw0 reads raw residual; row space in raw coords is directly (V*lw)^T
for h in range(6):
    q,k,v,o=headmats(0,h)   # v already includes lw0 -> acts on raw normalized residual
    Uo,So,Vo=np.linalg.svd(o@v); 
    R=Vo[:40].T
    c=np.linalg.svd(D[:,:26].T@R,compute_uv=False)
    print('L0H%d OV-in(40) vs D26'%h,c.round(2))
c=np.linalg.svd(D[:,:30].T@np.linalg.svd(headmats(0,3)[0].T@headmats(0,3)[1])[2][:24].T,compute_uv=False); print('D30 vs L0H3 key space',c.round(2))
```