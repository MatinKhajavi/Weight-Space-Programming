```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
def headmats(l,h):
    lw=W[f'ln1_w.{l}']; s=slice(64*h,64*h+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
spec=[5, 18, 21, 23, 29, 39, 58, 73, 76, 97, 99, 153, 168, 184, 214, 215, 236, 266, 275, 319, 322, 364, 375, 382, 383]
F=W['W_fc.0']*W['ln2_w.0']; nf=np.linalg.norm(F,axis=1)
outs=[]
for h in range(6):
    q,k,v,o=headmats(0,h); U,S,Vt=np.linalg.svd(o@v); r=int((S>S[0]*0.05).sum()); outs.append(U[:,:min(r,40)])
Hout=np.linalg.svd(np.hstack(outs),full_matrices=False)[0]
Fa=F[nf>1.0]; Fr=Fa-(Fa@Hout)@Hout.T
U,S,Vt=np.linalg.svd(Fr,full_matrices=False); D=Vt[:30].T
print('special-dim energy per D',(D[spec]**2).sum(0).round(2),'expect',len(spec)/384)
print('ipr per D',(D**4).sum(0).round(3))
mask=np.ones(384,bool); mask[spec]=False
S2=np.linalg.svd(Fr[:,mask],compute_uv=False); print('masked sv',S2[:32].round(2))
# also: ones-vector (LN mean) direction
one=np.ones(384)/np.sqrt(384); print('ones energy per D',((one@D)**2).round(3))
lnb=W['ln2_b.0']; print('ln2_b.0 direction energy', ((lnb/np.linalg.norm(lnb))@D)**2)
```