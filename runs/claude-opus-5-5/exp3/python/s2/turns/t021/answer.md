```python
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
w0=W['ln1_w.0']; w2=W['ln2_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s[:k]
T,_=rs(np.vstack([W['W_v.0'][64:128]*w0, W['W_v.0'][320:384]*w0]),28)
nr=np.linalg.norm(W['W_fc.0'],axis=1); alive=nr>1
F=(W['W_fc.0']*w2)[alive]
G=F@T
fr=np.linalg.norm(G,axis=1)**2/np.linalg.norm(F,axis=1)**2
sel=fr>0.2
Gs=G[sel]/np.linalg.norm(G[sel],axis=1,keepdims=True)
C=Gs@Gs.T
print('n sel',sel.sum())
off=C[np.triu_indices(len(C),1)]
print('cos hist',np.histogram(off,bins=[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])[0])
Z=linkage(Gs,'average',metric='cosine')
for t in [0.2,0.3,0.4,0.5]:
    cl=fcluster(Z,t,'distance'); sizes=np.bincount(cl)[1:]
    print(t,'nclusters',len(sizes),'sizes>=3:',(sizes>=3).sum(), sorted(sizes)[::-1][:40])
```