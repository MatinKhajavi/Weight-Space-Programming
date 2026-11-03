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
vow=[12,13,16,21,25]; cons=[c for c in range(26) if c not in vow]; sp=26
def blocks(B,name):
    B=B-B.mean(1,keepdims=True)
    def m(r,c):
        vals=[B[i,j] for i in r for j in c if i!=j]; return np.mean(vals)
    print(f'{name}: VV {m(vow,vow):+.3f} VC {m(vow,cons):+.3f} CV {m(cons,vow):+.3f} CC {m(cons,cons):+.3f} | diag mean {np.mean(np.diag(B)[:27]):+.3f} | sp->sp {B[sp,sp]:+.3f} V->sp {m(vow,[sp]):+.3f} C->sp {m(cons,[sp]):+.3f} sp->V {m([sp],vow):+.3f} sp->C {m([sp],cons):+.3f}')
for heads in [[(3,5)],[(3,1)],[(3,5),(3,1)],[(2,1)],[(3,0)]]:
    Ou=sum([(OV(l,j)@D.T).T/np.linalg.norm((OV(l,j)@D.T).T,axis=1,keepdims=True) for (l,j) in heads])
    Ou=Ou-Ou[:27].mean(0)
    B=D[:27]@Ou[:27].T
    blocks(B,str(heads)+' direct(t_cur . o_next)')
    # prev-token path: OV0(5) t_prev . o_next  (skip-trigram-ish from previous token)
    B2=(OV(0,5)@D[:27].T).T@Ou[:27].T
    blocks(B2,str(heads)+' viaL0H5')
```