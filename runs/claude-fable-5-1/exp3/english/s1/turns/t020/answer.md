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
F = W["W_fc.0"]*W["ln2_w.0"]
OVall = np.hstack([OVe[(0,j)] for j in range(6)])
u,s,_ = np.linalg.svd(OVall,full_matrices=False); Uo=u[:,:100]
Fcomp = F - (F@Uo)@Uo.T
sv = np.linalg.svd(Fcomp,compute_uv=False)
print("Fcomp sv 20..40:", sv[20:40].round(2))
for dimE in [28,32,36]:
    E = rsv(Fcomp,dimE)
    FE = F@E; nrm=np.linalg.norm(FE,axis=1); frac = nrm**2/(np.linalg.norm(F,axis=1)**2)
    for thr in [0.4,0.5]:
        sel = np.where(frac>thr)[0]
        X = FE[sel]/nrm[sel][:,None]
        D = 1-np.abs(X@X.T); D=(D+D.T)/2; np.fill_diagonal(D,0)
        Z = linkage(squareform(D,checks=False), method='average')
        for t in [0.4,0.5]:
            lab = fcluster(Z, t=t, criterion='distance')
            cents=[]; szs=[]
            for c in range(1,lab.max()+1):
                idx=np.where(lab==c)[0]
                if len(idx)<2: continue
                V=X[idx]; sg=np.sign(V@V[0]); V=V*sg[:,None]; m=V.mean(0); m/=np.linalg.norm(m); cents.append(m); szs.append(len(idx))
            C=np.array(cents)
            # merge centroids with |cos|>0.5 greedily via clustering on centroids
            Dc = 1-np.abs(C@C.T); np.fill_diagonal(Dc,0); Dc=(Dc+Dc.T)/2
            Zc = linkage(squareform(Dc,checks=False), method='complete')
            labc = fcluster(Zc, t=0.5, criterion='distance')
            # rank of centroid matrix
            cs = np.linalg.svd(C,compute_uv=False)
            print(f"dimE={dimE} thr={thr} t={t}: clusters {len(szs)} -> merged {labc.max()}; centroid sv (count>0.3*max): {(cs>0.3*cs.max()).sum()}; smallest cluster sizes {sorted(szs)[:6]}")
```