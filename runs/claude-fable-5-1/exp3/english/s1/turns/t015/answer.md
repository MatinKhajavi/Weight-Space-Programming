```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
def eff(nm,j,l=0): 
    s=slice(64*j,64*(j+1)); return W[nm][s]*W[f"ln1_w.{l}"]
def rsv(A,r):
    _,_,vt=np.linalg.svd(A,full_matrices=False); return vt[:r].T
def lsv(A,r):
    u,_,_=np.linalg.svd(A,full_matrices=False); return u[:,:r]
OVe = {(l,j): W[f"W_o.{l}"][:,64*j:64*(j+1)]@eff(f"W_v.{l}",j,l) for l in range(4) for j in range(6)}
F = W["W_fc.0"]*W["ln2_w.0"]; w2=W["ln2_w.0"]
OVall = np.hstack([OVe[(0,j)] for j in range(6)])
u,s,_ = np.linalg.svd(OVall,full_matrices=False); Uo=u[:,:100]
Fcomp = F - (F@Uo)@Uo.T
E = rsv(Fcomp,28)
U2 = lsv(OVe[(0,2)],28)
def cluster_dirs(S, thr_frac=0.5, t=0.4):
    FS = F@S; nrm=np.linalg.norm(FS,axis=1); frac = nrm**2/(np.linalg.norm(F,axis=1)**2)
    sel = frac>thr_frac
    X = FS[sel]/nrm[sel][:,None]
    D = 1-np.abs(X@X.T); D=(D+D.T)/2; np.fill_diagonal(D,0)
    Z = linkage(squareform(D,checks=False), method='average')
    lab = fcluster(Z, t=t, criterion='distance')
    cents=[]; szs=[]
    for c in range(1,lab.max()+1):
        idx=np.where(lab==c)[0]
        if len(idx)<2: continue
        V=X[idx]; sg=np.sign(V@V[0]); V=V*sg[:,None]; m=V.mean(0); m/=np.linalg.norm(m); cents.append(m); szs.append(len(idx))
    return S@np.array(cents).T, np.array(szs), sel.sum()
Cfull, szE, nE = cluster_dirs(E,0.5,0.4)
print("E clusters:", Cfull.shape[1], "from", nE, "neurons; sizes", szE)
for thr in [0.3,0.4,0.5]:
    Dfull, szU, nU = cluster_dirs(U2,thr,0.4)
    print(f"U2 clusters (frac>{thr}):", Dfull.shape[1], "from", nU, "neurons; sizes", szU)
Dfull, szU, nU = cluster_dirs(U2,0.3,0.4)
# map token directions through L0H2: OVe2 applied to z_t ~ c_t / w2
z = Cfull/np.clip(w2,0.2,None)[:,None]
img = OVe[(0,2)]@z; img/=np.linalg.norm(img,axis=0,keepdims=True)
M = np.abs(Dfull.T@img)    # U2 clusters x E clusters
print("matching matrix max per E-token:", M.max(0).round(2))
print("matching matrix max per U2-cluster:", M.max(1).round(2))
print("fraction of E tokens whose best U2 match is also that U2 cluster's best E token:", np.mean([M[:,t].argmax()==c and M[c].argmax()==t for t,c in enumerate(M.argmax(0))]))
# also check: is the image of the E-token set 'spread' (distinct directions)? pairwise cos of img
print("pairwise |cos| of OVe2-images of token dirs: max per token", np.sort(np.abs(img.T@img-np.eye(img.shape[1])).max(1))[::-1][:10].round(2))
```