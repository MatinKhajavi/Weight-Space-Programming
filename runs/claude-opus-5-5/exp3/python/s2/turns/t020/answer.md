```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
w0=W['ln1_w.0']; w2=W['ln2_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s[:k]
T,_=rs(np.vstack([W['W_v.0'][64:128]*w0, W['W_v.0'][320:384]*w0]),28)
nr=np.linalg.norm(W['W_fc.0'],axis=1); alive=nr>1
F=(W['W_fc.0']*w2)[alive]
G=F@T
print('alive',alive.sum(),'G sv',np.round(np.linalg.svd(G,compute_uv=False),1))
fr=np.linalg.norm(G,axis=1)**2/np.linalg.norm(F,axis=1)**2
print('frac in T alive', np.round(np.quantile(fr,[0.1,0.5,0.9]),3))
for nc in [20,24,28]:
    ica=FastICA(n_components=nc,random_state=0,max_iter=5000,whiten='unit-variance')
    S=ica.fit_transform(G)
    k=kurtosis(S,axis=0)
    print(nc,'kurt',np.round(np.sort(k)[::-1],1))
```