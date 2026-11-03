```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
from scipy.cluster.hierarchy import linkage, fcluster
def eff(nm,j,l=0): 
    s=slice(64*j,64*(j+1)); return W[nm][s]*W[f"ln1_w.{l}"]
def rsv(A,r):
    _,_,vt=np.linalg.svd(A,full_matrices=False); return vt[:r].T
OVe = {(l,j): W[f"W_o.{l}"][:,64*j:64*(j+1)]@eff(f"W_v.{l}",j,l) for l in range(4) for j in range(6)}
F = W["W_fc.0"]*W["ln2_w.0"]
OVall = np.hstack([OVe[(0,j)] for j in range(6)])
u,s,_ = np.linalg.svd(OVall,full_matrices=False); Uo=u[:,:100]
Fcomp = F - (F@Uo)@Uo.T
for dimE in [28,34]:
    E = rsv(Fcomp,dimE)
    FE = F@E; nrm=np.linalg.norm(FE,axis=1); fracE = nrm**2/(np.linalg.norm(F,axis=1)**2)
    for thr in [0.4,0.5]:
        sel = fracE>thr
        X = FE[sel]/nrm[sel][:,None]
        Z = linkage(X, method='average', metric='cosine')
        for t in [0.3,0.4,0.5]:
            lab = fcluster(Z, t=t, criterion='distance')
            sizes = np.bincount(lab)[1:]
            print(f"dimE={dimE} sel>{thr} (n={sel.sum()}) cos-dist thr {t}: {len(sizes)} clusters; clusters with >=2: {(sizes>=2).sum()}, >=3: {(sizes>=3).sum()}; sizes: {np.sort(sizes)[::-1][:35]}")
```