```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
w0=W['ln1_w.0']; w2=W['ln2_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s[:k]
def OV(l,j):
    w=W[f'ln1_w.{l}']; sl=slice(64*j,64*j+64); return W[f'W_o.{l}'][:,sl]@(W[f'W_v.{l}'][sl]*w)
T,_=rs(np.vstack([W['W_v.0'][64:128]*w0, W['W_v.0'][320:384]*w0]),40)
P,_=rs(np.vstack([W['W_q.0'][64:128]*w0, W['W_q.0'][320:384]*w0, W['W_k.0'][64:128]*w0, W['W_k.0'][320:384]*w0, W['W_v.0'][256:320]*w0,W['W_v.0'][0:64]*w0]),24)
Tp=T-P@(P.T@T); U,s,_=np.linalg.svd(Tp,full_matrices=False); Tp=U[:,:28]
alive=np.linalg.norm(W['W_fc.0'],axis=1)>1
F=(W['W_fc.0']*w2)[alive]
G=F@Tp; Gp=F@OV(0,5)@Tp; G1=F@OV(0,1)@Tp
res=[]
for seed in range(4):
    ica=FastICA(n_components=28,random_state=seed,max_iter=10000,whiten='unit-variance',fun='logcosh',tol=1e-5)
    S=ica.fit_transform(G); sk=np.sign(((S-S.mean(0))**3).mean(0))
    E=ica.components_*sk[:,None]; E/=np.linalg.norm(E,axis=1,keepdims=True)
    res.append(E)
    kc=kurtosis(G@E.T,axis=0); kp=kurtosis(Gp@E.T,axis=0); kr=kurtosis(Gp@np.linalg.qr(np.random.randn(28,28))[0],axis=0)
    print(seed,'cur kurt',np.round(np.sort(kc)[::-1]).astype(int))
    print('   prev kurt along same dirs',np.round(np.sort(kp)[::-1]).astype(int), ' random-dir prev kurt', np.round(np.sort(kr)[::-1][:6]).astype(int))
# consistency across seeds
for s2 in range(1,4):
    C=np.abs(res[0]@res[s2].T); print('seed0 vs',s2,'matched cos',np.round(np.sort(C.max(1)),2))
```