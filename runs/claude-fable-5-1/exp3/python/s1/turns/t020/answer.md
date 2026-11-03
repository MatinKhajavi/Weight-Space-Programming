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
kT=26
T=U[:,::-1][:,:kT]; PT=T@T.T
q0,k0,v0,o0=head(0,0); u,s,vt=np.linalg.svd(k0@T,full_matrices=False); A=T@vt[0]
q3,k3,v3,o3=head(0,3); u,s,vt=np.linalg.svd(k3@T,full_matrices=False); B=T@vt[0]
B=B-A*(A@B); B/=np.linalg.norm(B)
# samples: rows of W_fc.0, W_fc.1, W_fc.2, W_fc.3 projected on T; also Q,K,V rows of all heads
X=np.vstack([W[f'W_fc.{l}'] for l in range(4)]+[W[f'W_{m}.{l}'] for l in range(4) for m in ['q','k','v']])@T
print('samples',X.shape)
# kurtosis along random directions vs along A,B
def kurt(z): z=(z-z.mean())/z.std(); return (z**4).mean()-3
rng=np.random.default_rng(0)
print('kurtosis along A:',kurt(X@(T.T@A)),' B:',kurt(X@(T.T@B)))
print('kurtosis random dirs:',np.round([kurt(X@(r/np.linalg.norm(r))) for r in rng.standard_normal((8,kT))],2))
for nc in [kT]:
    ica=FastICA(n_components=nc,random_state=0,max_iter=2000,tol=1e-4,whiten='unit-variance')
    S=ica.fit_transform(X)
    comps=ica.components_  # (nc, kT) in T coords (unmixing); mixing_ gives directions
    mix=ica.mixing_  # (kT, nc)
    dirs=mix/np.linalg.norm(mix,axis=0)
    ks=[kurt(S[:,i]) for i in range(nc)]
    order=np.argsort(ks)[::-1]
    print('ICA component kurtoses (sorted):',np.round(np.array(ks)[order],1))
    D=dirs[:,order]
    print('cos(A, comps):',np.round(np.abs(D.T@(T.T@A)),2))
    print('cos(B, comps):',np.round(np.abs(D.T@(T.T@B)),2))
    G=np.abs(D.T@D); np.fill_diagonal(G,0); print('max |cos| between comps:',np.round(G.max(1),2))
    # Rayleigh of Ssum along comps
    print('Ssum Rayleigh along comps:',np.round([ (T@D[:,i])@Ssum@(T@D[:,i]) for i in range(nc)],2))
```