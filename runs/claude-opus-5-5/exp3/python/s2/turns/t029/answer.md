```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
w0=W['ln1_w.0']; w2=W['ln2_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s[:k]
T,_=rs(np.vstack([W['W_v.0'][64:128]*w0, W['W_v.0'][320:384]*w0]),32)
nr=np.linalg.norm(W['W_fc.0'],axis=1); alive=nr>1
F=(W['W_fc.0']*w2)[alive]
G=F@T
for nc in [24,26,27,28,30]:
    best=None
    for seed in range(3):
        ica=FastICA(n_components=nc,random_state=seed,max_iter=5000,whiten='unit-variance',fun='cube')
        S=ica.fit_transform(G)
        # make heavy tail positive
        sk=np.sign(((S-S.mean(0))**3).mean(0)); S=S*sk
        k=kurtosis(S,axis=0)
        print(nc,seed,'kurt',np.round(np.sort(k)[::-1],0).astype(int))
```