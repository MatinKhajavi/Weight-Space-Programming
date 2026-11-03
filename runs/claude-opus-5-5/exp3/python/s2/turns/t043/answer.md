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
u3,s3,_=np.linalg.svd(W['W_proj.3'],full_matrices=False); L=u3[:,:30]
for (l,j) in [(3,5),(3,1),(3,0),(2,1),(3,3),(3,2),(3,4)]:
    M=OV(l,j)@Tp
    sv=np.linalg.svd(M,compute_uv=False)
    print(f'L{l}H{j} OV on tokens sv',np.round(sv[:30],2),' frac output in L30 %.2f'%(np.linalg.norm(L.T@M)**2/np.linalg.norm(M)**2))
```