```python
import numpy as np
from sklearn.decomposition import FastICA
w0=W['ln1_w.0']; w2=W['ln2_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s[:k]
def QK(l,j):
    w=W[f'ln1_w.{l}']; sl=slice(64*j,64*j+64)
    return (W[f'W_q.{l}'][sl]*w).T@(W[f'W_k.{l}'][sl]*w)
def OV0(j):
    sl=slice(64*j,64*j+64); return W['W_o.0'][:,sl]@(W['W_v.0'][sl]*w0)
T,_=rs(np.vstack([W['W_v.0'][64:128]*w0, W['W_v.0'][320:384]*w0]),32)
nr=np.linalg.norm(W['W_fc.0'],axis=1); alive=nr>1
F=(W['W_fc.0']*w2)[alive]
G=F@T
ica=FastICA(n_components=28,random_state=0,max_iter=5000,whiten='unit-variance',fun='cube')
S=ica.fit_transform(G)
sk=np.sign(((S-S.mean(0))**3).mean(0))
E=ica.components_*sk[:,None]; E=E/np.linalg.norm(E,axis=1,keepdims=True)
D=(T@E.T).T
for j in [0,2,3]:
    M=QK(0,j)
    MD=M@D.T
    u,s,vt=np.linalg.svd(MD)
    q=u[:,0]; sc=q@MD
    if np.abs(sc.max())<np.abs(sc.min()): sc=-sc
    o=np.argsort(-sc)
    print(f'H{j} key token scores (sv {s[:3].round(2)}):',' '.join(f'{c}:{sc[c]:+.2f}' for c in o))
    MQ=D@M; u,s,vt=np.linalg.svd(MQ); sc=u[:,0]*s[0]
    if np.abs(sc.max())<np.abs(sc.min()): sc=-sc
    o=np.argsort(-sc)
    print(f'   query token scores (sv {s[:3].round(2)}):',' '.join(f'{c}:{sc[c]:+.2f}' for c in o))
```