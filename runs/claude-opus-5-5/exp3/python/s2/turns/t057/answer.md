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
D=Temb@Tp.T; Dd=E@Tp.T
pos_rows=np.vstack([W['W_q.0'][64:128]*w0, W['W_q.0'][320:384]*w0, W['W_k.0'][64:128]*w0, W['W_k.0'][320:384]*w0, W['W_v.0'][256:320]*w0])
for kP in [30,40,60]:
    Pl,_=rs(pos_rows,kP)
    ov=np.linalg.norm(Pl.T@D.T,axis=0)**2; ovd=np.linalg.norm(Pl.T@Dd.T,axis=0)**2
    print(kP,'emb dirs overlap with big P: tok27 %.2f, tokens mean %.2f max %.2f'%(ov[27],ov[:27].mean(),ov[:27].max()),' | dual: tok27 %.2f mean %.2f max %.2f'%(ovd[27],ovd[:27].mean(),ovd[:27].max()))
# how strongly is each direction read by V of all L0 heads vs Q/K of L0 heads
for nm in ['W_q.0','W_k.0','W_v.0']:
    M=W[nm]*w0
    r=np.linalg.norm(M@D.T,axis=0)
    print(nm,'read norm tok27 %.2f  tokens mean %.2f min %.2f max %.2f'%(r[27],r[:27].mean(),r[:27].min(),r[:27].max()))
# per-head V read
for j in range(6):
    M=W['W_v.0'][64*j:64*j+64]*w0; r=np.linalg.norm(M@D.T,axis=0)
    print(' V head',j,'tok27 %.2f tokens mean %.2f'%(r[27],r[:27].mean()))
```