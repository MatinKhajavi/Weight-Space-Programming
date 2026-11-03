```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
from scipy.cluster.hierarchy import linkage, fcluster
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
FE = F@E; nrm=np.linalg.norm(FE,axis=1); fracE = nrm**2/(np.linalg.norm(F,axis=1)**2)
sel = fracE>0.5
X = FE[sel]/nrm[sel][:,None]
# cluster with sign-invariance: use |cos|
Xs = X.copy()
# flip signs to align with first principal direction? instead do linkage on 1-|cos|
D = 1-np.abs(X@X.T); D=(D+D.T)/2; np.fill_diagonal(D,0)
from scipy.spatial.distance import squareform
Z = linkage(squareform(D,checks=False), method='average')
lab = fcluster(Z, t=0.4, criterion='distance')
sizes=np.bincount(lab)[1:]
print("sign-invariant clusters:", len(sizes), "with>=2:", (sizes>=2).sum(), "sizes:", np.sort(sizes)[::-1])
# centroids with sign alignment
cents=[]
for c in range(1,len(sizes)+1):
    idx=np.where(lab==c)[0]
    if len(idx)<2: continue
    V=X[idx]
    # align signs to the first member
    sg=np.sign(V@V[0]); V=V*sg[:,None]
    m=V.mean(0); m/=np.linalg.norm(m); cents.append(m)
C = np.array(cents)   # k x 28 (in E coords)
k=len(C)
Cfull = E@C.T          # 384 x k, unit directions in residual space
print("num token candidates:", k, " centroid pairwise |cos| max:", np.sort(np.abs(Cfull.T@Cfull-np.eye(k)).max(1))[::-1][:8])
# ensure sign: a token direction should be the direction neurons mostly read positively? use the sign such that the sum of neuron readings (over E-dominant neurons) is positive
A = F@Cfull           # 1536 x k : current-token reading
sg = np.sign((np.maximum(A,0)**2).sum(0)-(np.maximum(-A,0)**2).sum(0)); Cfull*=sg; A=F@Cfull
z = Cfull/np.clip(w2,0.2,None)[:,None]     # pre-LN-weight direction estimates
Bprev = F@OVe[(0,2)]@z   # 1536 x k : previous-token reading via L0H2
print("per-token: #neurons with A>0.5*max :", (A>0.5*A.max(0)).sum(0))
print("per-token: strength sum relu(A)^2:", (np.maximum(A,0)**2).sum(0).round(1))
print("per-token: prev reading strength sum relu(B)^2:", (np.maximum(Bprev,0)**2).sum(0).round(1), " neg:", (np.maximum(-Bprev,0)**2).sum(0).round(1))
G = np.maximum(Bprev,0).T @ np.maximum(A,0)   # k x k, [prev, cur]
Gn = G/ G.sum()
# bipartiteness test
S = G+G.T; np.fill_diagonal(S,0)
ev,evec = np.linalg.eigh(S)
print("eigs of symmetrized bigram graph: max %.2f min %.2f ratio %.2f" % (ev[-1], ev[0], -ev[0]/ev[-1]))
part = evec[:,0]>0
cross = (G[part][:,~part].sum()+G[~part][:,part].sum())/G.sum()
print("partition sizes", part.sum(), (~part).sum(), " fraction of bigram mass crossing partition: %.2f" % cross, " prev-in-A->cur-in-B: %.2f, B->A: %.2f" % (G[part][:,~part].sum()/G.sum(), G[~part][:,part].sum()/G.sum()))
# also diagonal mass (same token bigram)
print("diagonal mass fraction: %.3f (uniform would be %.3f)" % (np.trace(G)/G.sum(), 1/k))
# entropy / sparsity of G rows
Gr = G/G.sum(1,keepdims=True)
H = -(Gr*np.log(Gr+1e-12)).sum(1)
print("row entropies (nats), max possible %.2f:" % np.log(k), np.sort(H).round(2))
```