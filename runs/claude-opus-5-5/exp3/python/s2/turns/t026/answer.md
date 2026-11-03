```python
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
w0=W['ln1_w.0']; w2=W['ln2_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s[:k]
P,_=rs(np.vstack([W['W_q.0'][64:128]*w0, W['W_q.0'][320:384]*w0, W['W_k.0'][64:128]*w0, W['W_k.0'][320:384]*w0]),20)
hi=np.argsort(-w0)[:80]
print('P mass on hi coords', ((P[hi]**2).sum(0)).round(2))
nr=np.linalg.norm(W['W_fc.0'],axis=1); alive=nr>1
F=(W['W_fc.0']*w2)[alive]
kT=36
T,_=rs(np.vstack([W['W_v.0'][64:128]*w0, W['W_v.0'][320:384]*w0]),kT)
print('T dirs mass on hi coords', ((T[hi]**2).sum(0)).round(2))
G=F@T
fr=np.linalg.norm(G,axis=1)**2/np.linalg.norm(F,axis=1)**2
sel=fr>0.2
Gs=G[sel]/np.linalg.norm(G[sel],axis=1,keepdims=True)
Z=linkage(Gs,'average',metric='cosine')
cl=fcluster(Z,0.45,'distance')
rows=[]
for c in np.unique(cl):
    m=cl==c
    if m.sum()>=3:
        v=Gs[m].mean(0); v/=np.linalg.norm(v); d=T@v
        rows.append((m.sum(), (d[hi]**2).sum(), np.linalg.norm(P.T@d)**2, np.mean(np.linalg.norm(G[sel][m],axis=1))))
for r in sorted(rows,key=lambda r:-r[1]): print('size %d  himass %.2f  inP %.2f  meanTnorm %.2f'%r)
```