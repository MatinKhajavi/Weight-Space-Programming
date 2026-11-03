```python
import numpy as np
from sklearn.decomposition import FastICA
np.set_printoptions(precision=2, suppress=True, linewidth=250)
d=384; hd=64; H=6
def head(l,j):
    s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s], W[f'W_k.{l}'][s], W[f'W_v.{l}'][s], W[f'W_o.{l}'][:,s]
OV={}
for l in range(4):
    for j in range(H):
        q,k,v,o=head(l,j); OV[(l,j)]=o@v
Ssum=np.zeros((d,d))
for (l2,h2) in [(2,1),(3,1),(3,5)]:
    q,k,v,o=head(l2,h2)
    M=(q.T@k)@OV[(0,5)]; S=(M+M.T)/2; Ssum+=S/np.linalg.norm(S)
ev,U=np.linalg.eigh(Ssum)
def kurt(z): z=(z-z.mean())/z.std(); return (z**4).mean()-3
Xfull=np.vstack([W[f'W_fc.{l}'] for l in range(4)])
Xv=np.vstack([W[f'W_v.{l}'] for l in range(4)])
for kT in [24,28,32,36]:
    T=U[:,::-1][:,:kT]
    for nm,Xs in [('fc',Xfull),('fc+v',np.vstack([Xfull,Xv]))]:
        X=Xs@T
        ica=FastICA(n_components=kT,random_state=0,max_iter=3000,tol=1e-4,whiten='unit-variance')
        S=ica.fit_transform(X)
        ks=np.sort([kurt(S[:,i]) for i in range(kT)])[::-1]
        print(f'kT={kT} {nm}: kurtoses {np.round(ks,1)}  n>3: {(ks>3).sum()} n>2: {(ks>2).sum()}')
# Also: ICA on the complement of T? check that kurtosis there is low (sanity)
T=U[:,::-1][:,:28]; PP=np.eye(d)-T@T.T
u,s,vt=np.linalg.svd(Xfull@PP,full_matrices=False)
Xc=(Xfull@PP)@vt[:28].T
ica=FastICA(n_components=28,random_state=0,max_iter=3000,tol=1e-4,whiten='unit-variance'); S=ica.fit_transform(Xc)
print('complement top-28 PCA dirs ICA kurtoses:',np.round(np.sort([kurt(S[:,i]) for i in range(28)])[::-1],1))
```