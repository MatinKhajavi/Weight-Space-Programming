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
def OV0(j):
    sl=slice(64*j,64*j+64); return W['W_o.0'][:,sl]@(W['W_v.0'][sl]*w0)
T,_=rs(np.vstack([W['W_v.0'][64:128]*w0, W['W_v.0'][320:384]*w0]),32)
nr=np.linalg.norm(W['W_fc.0'],axis=1); alive=nr>1
F=(W['W_fc.0']*w2)[alive]
G=F@T
ica=FastICA(n_components=28,random_state=0,max_iter=5000,whiten='unit-variance',fun='cube')
S=ica.fit_transform(G)
sk=np.sign(((S-S.mean(0))**3).mean(0))
E=ica.components_*sk[:,None]
E=E/np.linalg.norm(E,axis=1,keepdims=True)    # 28 x 32 token dirs in T coords
D=(T@E.T).T                                    # 28 x 384 raw residual dirs
np.save('/tmp/D.npy',D); np.save('/tmp/E.npy',E)
print('Gram of token dirs (abs offdiag max per row)',np.round(np.sort(np.abs(D@D.T-np.eye(28)),1)[:,-1],2))
R=G@E.T
k=kurtosis(R,axis=0)
nstrong=[(R[:,c]>3*R[:,c].std()).sum() for c in range(28)]
cp5=[np.linalg.norm(OV0(5)@D[c]) for c in range(28)]
cp1=[np.linalg.norm(OV0(1)@D[c]) for c in range(28)]
h0=[float(np.linalg.norm(QK(0,0)@D[c])) for c in range(28)]
h2=[float(np.linalg.norm(QK(0,2)@D[c])) for c in range(28)]
h3=[float(np.linalg.norm(QK(0,3)@D[c])) for c in range(28)]
ind=[float(D[c]@QK(3,5)@OV0(5)@D[c]) for c in range(28)]
ind2=[float(D[c]@QK(2,1)@OV0(5)@D[c]) for c in range(28)]
for c in np.argsort(-k):
    print(f'tok{c:2d} kurt {k[c]:5.0f} nstrong {nstrong[c]:3d} copy5 {cp5[c]:.2f} copy1 {cp1[c]:.2f} H0key {h0[c]:.3f} H2key {h2[c]:.3f} H3key {h3[c]:.3f} ind L3H5 {ind[c]:+.3f} L2H1 {ind2[c]:+.3f}')
```