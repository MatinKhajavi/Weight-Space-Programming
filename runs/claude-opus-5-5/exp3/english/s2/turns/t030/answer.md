```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def headmats(l,j):
    g=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
Up,sp,_=np.linalg.svd(W['W_proj.3'],full_matrices=False)
Wq,Wk,Wv,Wo=headmats(3,1); OV1=Wo@Wv
U1,s1,_=np.linalg.svd(OV1,full_matrices=False)
a,c,bt=np.linalg.svd(Up[:,:45].T@U1[:,:45])
S=Up[:,:45]@a[:,:28]
Wp=W['W_proj.3']           # 384 x 1536
X=S.T@Wp                   # 28 x 1536
nrm=np.linalg.norm(X,axis=0); tot=np.linalg.norm(Wp,axis=0)
print('neuron out norm in S: percentiles',np.percentile(nrm,[10,50,90,99]).round(2),' frac in S median %.2f'%np.median(nrm/tot))
idx=np.argsort(-nrm)[:600]
Y=X[:,idx]/nrm[idx]
C=Y.T@Y
np.fill_diagonal(C,0)
print('max |cos| distribution among top600:',np.percentile(np.abs(C).max(1),[10,25,50,75,90]).round(2))
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import pdist
D=pdist(Y.T,'cosine')
L=linkage(D,'average')
for t in [0.3,0.4,0.5,0.6]:
    lab=fcluster(L,t,'distance'); cnt=np.bincount(lab)
    big=np.sort(cnt)[::-1]
    print('thr',t,'nclusters',len(set(lab)),'sizes',big[:40])
```