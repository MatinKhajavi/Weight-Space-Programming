```python
import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
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
kc=kurtosis(G@E.T,axis=0); order=np.argsort(-kc); E=E[order]; kc=kc[order]
Temb=np.linalg.pinv(E).T; Temb/=np.linalg.norm(Temb,axis=1,keepdims=True)
for name,B in [('dual',E),('emb',Temb)]:
    Dd=B@Tp.T
    print(name,'gram offdiag max',np.round(np.abs(Dd@Dd.T-np.eye(28)).max(),2))
    cp5=np.array([np.linalg.norm(OV(0,5)@d) for d in Dd]); cp1=np.array([np.linalg.norm(OV(0,1)@d) for d in Dd])
    ind={k:np.array([d@QK(*k)@OV(0,5)@d for d in Dd]) for k in [(2,1),(3,1),(3,5)]}
    nstr=np.array([(np.abs(G@B[c])>3*np.std(G@B[c])).sum() for c in range(28)])
    rows=[]
    for c in range(28):
        rows.append((c,kc[c],cp5[c],cp1[c],ind[(2,1)][c],ind[(3,1)][c],ind[(3,5)][c],nstr[c]))
    for r in sorted(rows,key=lambda r:-r[2]):
        print('  tok%2d kurt%5.0f copy5 %.2f copy1 %.2f indL2H1 %+.3f indL3H1 %+.3f indL3H5 %+.3f nstrong %d'%r)
```