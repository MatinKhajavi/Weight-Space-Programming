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
Z=linkage(Gs,'average',metric='cosine')
cl=fcluster(Z,0.4,'distance')
cents=[];sz=[]
for c in np.unique(cl):
    m=cl==c
    if m.sum()>=3:
        v=Gs[m].mean(0); cents.append(v/np.linalg.norm(v)); sz.append(m.sum())
Cm=np.array(cents)
np.set_printoptions(linewidth=250)
print('sizes',sz)
print(np.round(Cm@Cm.T,1))
print('centroid sv',np.round(np.linalg.svd(Cm,compute_uv=False),2))
```