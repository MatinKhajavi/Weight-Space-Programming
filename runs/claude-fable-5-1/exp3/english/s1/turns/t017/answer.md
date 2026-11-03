```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
def eff(nm,j,l=0): 
    s=slice(64*j,64*(j+1)); return W[nm][s]*W[f"ln1_w.{l}"]
def rsv(A,r):
    _,_,vt=np.linalg.svd(A,full_matrices=False); return vt[:r].T
OVe = {(l,j): W[f"W_o.{l}"][:,64*j:64*(j+1)]@eff(f"W_v.{l}",j,l) for l in range(4) for j in range(6)}
F = W["W_fc.0"]*W["ln2_w.0"]; w2=W["ln2_w.0"]
OVall = np.hstack([OVe[(0,j)] for j in range(6)])
u,s,_ = np.linalg.svd(OVall,full_matrices=False); Uo=u[:,:100]
Fcomp = F - (F@Uo)@Uo.T
E = rsv(Fcomp,28)
FE = F@E; nrm=np.linalg.norm(FE,axis=1); frac = nrm**2/(np.linalg.norm(F,axis=1)**2)
sel = np.where(frac>0.5)[0]
X = FE[sel]/nrm[sel][:,None]
D = 1-np.abs(X@X.T); D=(D+D.T)/2; np.fill_diagonal(D,0)
Z = linkage(squareform(D,checks=False), method='average')
lab = fcluster(Z, t=0.4, criterion='distance')
cents=[]; sizes=[]
for c in range(1,lab.max()+1):
    idx=np.where(lab==c)[0]
    if len(idx)<2: continue
    V=X[idx]; sg=np.sign(V@V[0]); V=V*sg[:,None]; m=V.mean(0); m/=np.linalg.norm(m); cents.append(m); sizes.append(len(idx))
C = E@np.array(cents).T   # 384 x k
A = F@C
sg = np.sign((np.maximum(A,0)**2).sum(0)-(np.maximum(-A,0)**2).sum(0)); C*=sg
k=C.shape[1]
# z_t estimate (pre-LN-weight): C / w2 ; then the layer-1 LN1 input direction ~ z * ln1_w.1
z = C/np.clip(w2,0.2,None)[:,None]
h1 = z*W["ln1_w.1"][:,None]; h1/=np.linalg.norm(h1,axis=0,keepdims=True)
Btot=np.zeros((k,k))
for j in range(6):
    s=slice(64*j,64*(j+1))
    q=W["W_q.1"][s]; kk=W["W_k.1"][s]
    B = (q@h1).T@(kk@h1)/8    # query token x key token
    Btot+=B
print("layer-1 summed token-token logit form (rows=query token, cols=key token); diag mean %.3f offdiag mean %.3f offdiag std %.3f" % (np.diag(Btot).mean(), Btot[~np.eye(k,dtype=bool)].mean(), Btot[~np.eye(k,dtype=bool)].std()))
# structure: SVD of Btot
uu,ss,vv=np.linalg.svd(Btot)
print("sv:", ss[:8].round(2))
print("top left sv (query side):", uu[:,0].round(2))
print("top right sv (key side):", vv[0].round(2))
print("2nd left:", uu[:,1].round(2)); print("2nd right:", vv[1].round(2))
# cluster tokens by row profile
Zr = linkage(Btot, method='average', metric='correlation'); labr=fcluster(Zr,2,'maxclust'); print("row-profile 2 clusters sizes:", np.bincount(labr)[1:])
Zc = linkage(Btot.T, method='average', metric='correlation'); labc=fcluster(Zc,2,'maxclust'); print("col-profile 2 clusters sizes:", np.bincount(labc)[1:])
print("block means (row cluster x col cluster):")
for a in [1,2]:
    print([Btot[labr==a][:,labc==b].mean().round(3) for b in [1,2]])
print("cluster sizes (neurons):", sizes)
print("row cluster membership:", labr); print("col cluster membership:", labc)
```