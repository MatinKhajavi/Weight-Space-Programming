```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
w0=W['ln1_w.0']; w2=W['ln2_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s[:k]
def QK(l,j):
    w=W[f'ln1_w.{l}']; sl=slice(64*j,64*j+64)
    return (W[f'W_q.{l}'][sl]*w).T@(W[f'W_k.{l}'][sl]*w)
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
D=Temb@Tp.T
np.set_printoptions(linewidth=220,precision=2,suppress=True)
for l in [0,1,2,3]:
    for j in range(6):
        A=D@QK(l,j)@D.T     # query tok x key tok
        Ac=A-A.mean(1,keepdims=True)   # per-query centering (softmax invariant)
        d=np.diag(Ac)[:27]
        print(f'L{l}H{j} self-pair (centered) mean {d.mean():+.3f}  space(26) {d[26]:+.3f}  min tok {np.argmin(d)} {d.min():+.3f}  max tok {np.argmax(d)} {d.max():+.3f}')
```