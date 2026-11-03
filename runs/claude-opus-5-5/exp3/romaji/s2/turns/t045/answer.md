```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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
X=Fa@D; n=np.linalg.norm(X,axis=1); print('proj norm pct',np.percentile(n,[10,50,90]).round(2))
Xn=X/n[:,None]
G=Xn@Xn.T; np.fill_diagonal(G,0)
print('fraction of neurons with a partner cos>0.9: %.2f, >0.8: %.2f'%((G.max(1)>0.9).mean(),(G.max(1)>0.8).mean()))
for k in [12,16,20,22,24,26,28,30,34,40]:
    km=KMeans(k,n_init=10,random_state=0).fit(Xn); print(k,'inertia %.1f sil %.3f'%(km.inertia_,silhouette_score(Xn,km.labels_,metric='cosine')))
```