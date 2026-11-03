```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
w0=W['ln1_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s[:k]
def OV0(j):
    sl=slice(64*j,64*j+64); return W['W_o.0'][:,sl]@(W['W_v.0'][sl]*w0)
T,_=rs(np.vstack([W['W_v.0'][64:128]*w0, W['W_v.0'][320:384]*w0]),32)
samples={}
for l in range(4):
    Fl=W[f'W_fc.{l}']*W[f'ln2_w.{l}']
    alive=np.linalg.norm(W[f'W_fc.{l}'],axis=1)>1
    Fl=Fl[alive]
    samples[f'cur{l}']=Fl@T
    samples[f'prev{l}']=Fl@OV0(5)@T
    samples[f'prev1_{l}']=Fl@OV0(1)@T
for k,v in samples.items():
    n=np.linalg.norm(v,axis=1)
    print(k, v.shape[0], 'mean norm',n.mean().round(2))
def ica_kurt(X,nc):
    X=X/np.linalg.norm(X,axis=1,keepdims=True).mean()
    ica=FastICA(n_components=nc,random_state=0,max_iter=5000,whiten='unit-variance',fun='cube')
    S=ica.fit_transform(X); sk=np.sign(((S-S.mean(0))**3).mean(0)); S=S*sk
    return np.round(np.sort(kurtosis(S,axis=0))[::-1],0).astype(int)
for k in ['cur1','prev0','prev1','cur2']:
    print(k, ica_kurt(samples[k],30))
```