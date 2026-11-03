```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=250)
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
U,S,Vt=np.linalg.svd(Fr,full_matrices=False); D=Vt[:32].T
print('sv',S[:32].round(2))
for h in range(6):
    q,k,v,o=headmats(0,h)
    print('L0H%d OV gain'%h,np.linalg.norm(o@v@D,axis=0).round(2))
    print('     Vq gain',np.linalg.norm(q@D,axis=0).round(2))
    print('     Vk gain',np.linalg.norm(k@D,axis=0).round(2))
# random direction baseline
rng=np.random.default_rng(0); Rr=np.linalg.qr(rng.standard_normal((384,8)))[0]
for h in [0,3]:
    q,k,v,o=headmats(0,h); print('rand OV gain L0H%d'%h,np.linalg.norm(o@v@Rr,axis=0).round(2),'q',np.linalg.norm(q@Rr,axis=0).round(2))
```