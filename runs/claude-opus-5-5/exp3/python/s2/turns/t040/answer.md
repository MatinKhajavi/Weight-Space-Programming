```python
import numpy as np
from sklearn.decomposition import FastICA
w0=W['ln1_w.0']; w2=W['ln2_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s[:k]
def OV(l,j):
    w=W[f'ln1_w.{l}']; sl=slice(64*j,64*j+64); return W[f'W_o.{l}'][:,sl]@(W[f'W_v.{l}'][sl]*w)
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
from scipy.stats import kurtosis
kc=kurtosis(G@E.T,axis=0); order=np.argsort(-kc); E=E[order]; kc=kc[order]
D=E@Tp.T   # 28 x 384
print('token dir gram offdiag max',np.round(np.abs(D@D.T-np.eye(28)).max(),2))
cp5=np.array([np.linalg.norm(OV(0,5)@D[c]) for c in range(28)])
print('copy5',np.round(cp5,2))
for j in [0,1,2,3,5]:
    M=QK(0,j); MD=M@D.T; u,s,vt=np.linalg.svd(MD); q=u[:,0]; sc=q@MD
    if abs(sc.max())<abs(sc.min()): sc=-sc
    print(f'L0H{j} key tok (sv{np.round(s[:3],2)}):',' '.join(f'{c}:{sc[c]:+.2f}' for c in np.argsort(-sc)[:10]),' ... ',' '.join(f'{c}:{sc[c]:+.2f}' for c in np.argsort(-sc)[-5:]))
```