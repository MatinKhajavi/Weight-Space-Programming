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
G=F@Tp; Gp=F@OV(0,5)@Tp; G1=F@OV(0,1)@Tp
ica=FastICA(n_components=28,random_state=0,max_iter=10000,whiten='unit-variance',fun='logcosh',tol=1e-5)
S=ica.fit_transform(G); sk=np.sign(((S-S.mean(0))**3).mean(0))
E=ica.components_*sk[:,None]; E/=np.linalg.norm(E,axis=1,keepdims=True)
kc=kurtosis(G@E.T,axis=0); order=np.argsort(-kc); E=E[order]
Temb=np.linalg.pinv(E).T; Temb/=np.linalg.norm(Temb,axis=1,keepdims=True)
R=G@Temb.T; Rp=Gp@Temb.T; R1=G1@Temb.T
np.save('/tmp/x.npy',R)
for tok in [26,18,21,0,2]:
    rz=(R[:,tok]-R.mean(1))/R.std(1)
    top=np.argsort(-rz)[:15]
    print(f'== neurons detecting cur={tok} (z>{rz[top[-1]]:.1f})')
    # prev responses: rank of prev=tok among 28 and magnitude ratio
    for nm,X in [('H5',Rp),('H1',R1)]:
        mag_self=np.abs(X[top,tok]); mag_all=np.abs(X[top]).mean(1)
        sign_self=X[top,tok]
        z=(X[top,tok]-X[top].mean(1))/X[top].std(1)
        print(f'   prev via {nm}: |self|/mean|all| = {np.mean(mag_self/mag_all):.2f}   z(self) mean {z.mean():+.2f}  z list {np.round(z,1)}')
```