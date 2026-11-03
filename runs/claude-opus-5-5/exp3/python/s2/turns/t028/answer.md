```python
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
w0=W['ln1_w.0']; w2=W['ln2_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s[:k]
def QK(l,j):
    w=W[f'ln1_w.{l}']; sl=slice(64*j,64*j+64)
    return (W[f'W_q.{l}'][sl]*w).T@(W[f'W_k.{l}'][sl]*w)
def OV0(j):
    sl=slice(64*j,64*j+64); return W['W_o.0'][:,sl]@(W['W_v.0'][sl]*w0)
T,_=rs(np.vstack([W['W_v.0'][64:128]*w0, W['W_v.0'][320:384]*w0]),32)
nr=np.linalg.norm(W['W_fc.0'],axis=1); alive=nr>1
F=(W['W_fc.0']*w2)[alive]
G=F@T
fr=np.linalg.norm(G,axis=1)**2/np.linalg.norm(F,axis=1)**2
sel=fr>0.2
Gs=G[sel]/np.linalg.norm(G[sel],axis=1,keepdims=True)
Z=linkage(Gs,'average',metric='cosine')
cl=fcluster(Z,0.45,'distance')
cents=[];sizes=[]
for c in np.unique(cl):
    m=cl==c
    if m.sum()>=3:
        v=Gs[m].mean(0); cents.append(v/np.linalg.norm(v)); sizes.append(m.sum())
C=np.array(cents)
np.save('/tmp/cents.npy',C)
for j in [0,2,3]:
    M=QK(0,j)
    u,s,vt=np.linalg.svd(M@T); k=vt[0]; k2=vt[1]
    cs=C@k; cs2=C@k2
    o=np.argsort(-np.abs(cs))[:4]; o2=np.argsort(-np.abs(cs2))[:4]
    print(f'H{j} key tok dir1 (sv {s[0]:.2f}) top centroid cos',[(int(i),round(float(cs[i]),2),int(sizes[i])) for i in o], ' dir2 (sv %.2f)'%s[1],[(int(i),round(float(cs2[i]),2)) for i in o2])
```