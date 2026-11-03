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
G=F@Tp
ica=FastICA(n_components=28,random_state=0,max_iter=10000,whiten='unit-variance',fun='logcosh',tol=1e-5)
S=ica.fit_transform(G); sk=np.sign(((S-S.mean(0))**3).mean(0))
E=ica.components_*sk[:,None]; E/=np.linalg.norm(E,axis=1,keepdims=True)
kc=kurtosis(G@E.T,axis=0); order=np.argsort(-kc); E=E[order]
Temb=np.linalg.pinv(E).T; Temb/=np.linalg.norm(Temb,axis=1,keepdims=True)
Q=np.eye(384)-Tp@Tp.T
R=G@Temb.T
rz=(R-R.mean(1,keepdims=True))/R.std(1,keepdims=True)
X5=(F@Q@OV(0,5)@Tp)@Temb.T; X1=(F@Q@OV(0,1)@Tp)@Temb.T
np.set_printoptions(linewidth=220,precision=2,suppress=True)
# matrix: for cur token c (rows), sensitivity to prev token p (cols): mean |X[n,p]|/mean|X[n,:]| over top-12 neurons of c
for nm,X in [('H5',X5),('H1',X1)]:
    Msens=np.zeros((27,27))
    for c in range(27):
        top=np.argsort(-rz[:,c])[:12]
        A=np.abs(X[top][:,:27]); Msens[c]=(A/A.mean(1,keepdims=True)).mean(0)
    print(nm,'sensitivity to prev=26 for each cur token:',np.round(Msens[:,26],2))
    print(nm,'mean sensitivity per prev token (col means):',np.round(Msens.mean(0),2))
```