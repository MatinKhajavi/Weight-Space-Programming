```python
import numpy as np
from sklearn.decomposition import FastICA
w0=W['ln1_w.0']; w2=W['ln2_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s[:k]
T,_=rs(np.vstack([W['W_v.0'][64:128]*w0, W['W_v.0'][320:384]*w0]),28)
F=W['W_fc.0']*w2
G=F@T
print('G sv',np.round(np.linalg.svd(G,compute_uv=False),1))
from scipy.stats import kurtosis
ica=FastICA(n_components=28,random_state=0,max_iter=2000,whiten='unit-variance')
S=ica.fit_transform(G)   # neurons x comps
k=kurtosis(S,axis=0)
print('kurtosis of components',np.round(np.sort(k)[::-1],1))
```