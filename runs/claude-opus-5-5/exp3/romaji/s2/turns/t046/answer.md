```python
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
np.set_printoptions(precision=2, suppress=True, linewidth=250)
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
X=Fa@D; n=np.linalg.norm(X,axis=1); Xn=X/n[:,None]
Z=linkage(Xn,method='average',metric='cosine')
for t in [0.1,0.15,0.2,0.25,0.3,0.4]:
    lab=fcluster(Z,t,criterion='distance'); cnt=np.bincount(lab)[1:]
    print(t,'clusters',len(cnt),'with>=3 members',(cnt>=3).sum(),'with>=5',(cnt>=5).sum(),'sizes top',np.sort(cnt)[::-1][:40])
```