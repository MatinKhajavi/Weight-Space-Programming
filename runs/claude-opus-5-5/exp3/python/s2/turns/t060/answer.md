```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
w0=W['ln1_w.0']; w2=W['ln2_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s[:k]
def cs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return u[:,:k]
def OV(l,j):
    w=W[f'ln1_w.{l}']; sl=slice(64*j,64*j+64); return W[f'W_o.{l}'][:,sl]@(W[f'W_v.{l}'][sl]*w)
alive=np.linalg.norm(W['W_fc.0'],axis=1)>1
F=(W['W_fc.0']*w2)[alive]
P,_=rs(np.vstack([W['W_q.0'][64:128]*w0, W['W_q.0'][320:384]*w0, W['W_k.0'][64:128]*w0, W['W_k.0'][320:384]*w0, W['W_v.0'][256:320]*w0,W['W_v.0'][0:64]*w0]),24)
Os=np.hstack([cs(OV(0,j),20 if j in (1,5) else 10) for j in range(6)])
B=np.hstack([P,Os]); Qb,_=np.linalg.qr(B)
Fc=F-(F@Qb)@Qb.T
u,s,vt=np.linalg.svd(Fc,full_matrices=False)
print('Fc sv',np.round(s[:36],2))
for k in [30,32,34]:
    Tm=vt[1:k].T     # drop top (bias-like) direction
    G=F@Tm
    for nc in [k-1]:
        ica=FastICA(n_components=nc,random_state=0,max_iter=10000,whiten='unit-variance',fun='logcosh',tol=1e-5)
        S=ica.fit_transform(G); sk=np.sign(((S-S.mean(0))**3).mean(0)); S=S*sk
        kk=np.sort(kurtosis(S,axis=0))[::-1]
        print(k,nc,'kurtosis',np.round(kk).astype(int), ' n>=8:',(kk>=8).sum())
# overlap between this MLP0-derived token space and the V-derived Tp
T,_=rs(np.vstack([W['W_v.0'][64:128]*w0, W['W_v.0'][320:384]*w0]),40)
Tp=T-P@(P.T@T); U,s2,_=np.linalg.svd(Tp,full_matrices=False); Tp=U[:,:28]
print('principal cos between MLP0-space(top 29 after bias) and Tp28:',np.round(np.linalg.svd(vt[1:30]@Tp,compute_uv=False),2))
```