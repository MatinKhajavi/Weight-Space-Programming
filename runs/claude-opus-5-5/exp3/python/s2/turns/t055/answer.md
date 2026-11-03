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
D=Temb@Tp.T   # 28 x 384 token embedding dirs (normalized coords)
# output directions for each token from induction heads
O=[]
for (l,j) in [(3,5),(3,1)]:
    Oc=(OV(l,j)@D.T).T   # 28 x 384
    O.append(Oc/np.linalg.norm(Oc,axis=1,keepdims=True))
# consistency between the two heads' output dirs
C=O[0]@O[1].T
print('cos of token output dirs between L3H5 and L3H1: diag mean %.2f offdiag mean %.2f'%(np.mean(np.diag(C)[:27]), (C[:27,:27].sum()-np.trace(C[:27,:27]))/(27*26)))
Ou=(O[0]+O[1]); Ou=Ou-Ou[:27].mean(0); Ou/=np.linalg.norm(Ou,axis=1,keepdims=True)
B=D[:27]@Ou[:27].T     # cur token x next token (direct path proxy)
B=B-B.mean(1,keepdims=True)
np.set_printoptions(linewidth=250,precision=0,suppress=True)
vow=[12,13,16,21,25]
print('B*100 (rows=current, cols=next); vowels',vow,'space 26')
print((B*100).astype(int))
```