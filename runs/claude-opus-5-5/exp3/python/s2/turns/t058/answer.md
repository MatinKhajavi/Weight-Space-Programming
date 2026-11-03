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
D=Temb@Tp.T
feats_k=[];feats_q=[];feats_v=[]
for l in range(4):
    w=W[f'ln1_w.{l}']
    for j in range(6):
        sl=slice(64*j,64*j+64)
        feats_k.append((W[f'W_k.{l}'][sl]*w)@D.T)   # 64 x 28
        feats_q.append((W[f'W_q.{l}'][sl]*w)@D.T)
# also MLP reads in layers 1-3 (direct token reading)
for l in [1,2,3]:
    feats_v.append((W[f'W_fc.{l}']*W[f'ln2_w.{l}'])@D.T)
def sim(feats):
    X=np.vstack(feats)   # dims x 28
    X=X-X.mean(1,keepdims=True)
    X=X/np.linalg.norm(X,axis=0,keepdims=True)
    return X.T@X
np.set_printoptions(linewidth=250,precision=2,suppress=True)
for nm,f in [('keys',feats_k),('queries',feats_q),('laterMLP',feats_v),('all',feats_k+feats_q+feats_v)]:
    Sm=sim(f)
    o=np.argsort(-Sm[26])
    print(nm,'nearest to space(26):',[(int(c),round(float(Sm[26,c]),2)) for c in o[1:6]], ' nearest to 27:',[(int(c),round(float(Sm[27,c]),2)) for c in np.argsort(-Sm[27])[1:5]])
Sm=sim(feats_k+feats_q+feats_v)
# hierarchical clustering of tokens
from scipy.cluster.hierarchy import linkage, dendrogram
Z=linkage(1-Sm[np.triu_indices(28,1)],'average')
print(dendrogram(Z,no_plot=True)['ivl'])
```