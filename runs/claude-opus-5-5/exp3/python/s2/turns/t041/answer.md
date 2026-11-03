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
u,s,vt=np.linalg.svd(F,full_matrices=False)
v0=vt[0]*np.sign(u[:,0].sum())
print('frac of v0 in Tp',np.linalg.norm(Tp.T@v0)**2, ' in P',np.linalg.norm(P.T@v0)**2)
# mean neuron weight direction
m=F.mean(0); m/=np.linalg.norm(m); print('cos(mean F row, v0)',m@v0)
for name,vec in [('v0',v0),('meanF',m)]:
    t=Tp.T@vec   # 28 coords
    coef=np.linalg.lstsq(E.T,t,rcond=None)[0]   # t = sum coef_c E_c
    o=np.argsort(-coef)
    print(name,'token coefs:',' '.join(f'{c}:{coef[c]:+.2f}' for c in o))
    print('   projections E@t:',' '.join(f'{c}:{(E@t)[c]:+.2f}' for c in np.argsort(-(E@t))))
```