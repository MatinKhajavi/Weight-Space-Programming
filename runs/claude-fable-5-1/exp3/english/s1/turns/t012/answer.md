```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def eff(nm,j,l=0): 
    s=slice(64*j,64*(j+1)); return W[nm][s]*W[f"ln1_w.{l}"]
def rsv(A,r):
    _,_,vt=np.linalg.svd(A,full_matrices=False); return vt[:r].T
OVe = {(l,j): W[f"W_o.{l}"][:,64*j:64*(j+1)]@eff(f"W_v.{l}",j,l) for l in range(4) for j in range(6)}
F = W["W_fc.0"]*W["ln2_w.0"]
OVall = np.hstack([OVe[(0,j)] for j in range(6)])
u,s,_ = np.linalg.svd(OVall,full_matrices=False); Uo=u[:,:100]
Fcomp = F - (F@Uo)@Uo.T
E = rsv(Fcomp,30)
FE = F@E          # 1536 x 30
nrm = np.linalg.norm(FE,axis=1)
fracE = nrm**2/ (np.linalg.norm(F,axis=1)**2)
print("fraction of neuron input energy in E: quantiles", np.quantile(fracE,[0.1,0.25,0.5,0.75,0.9]))
sel = fracE>0.5
print("neurons with >50% energy in E:", sel.sum())
X = FE[sel]/nrm[sel][:,None]
C = X@X.T
iu = np.triu_indices(len(X),1)
h,_=np.histogram(C[iu],bins=np.linspace(-1,1,21))
print("pairwise cos histogram (E-projections of E-dominant neurons):", h)
# how many neurons have a near-duplicate (cos>0.9)?
print("neurons with some partner cos>0.9:", ((C-np.eye(len(X))).max(1)>0.9).sum(), " >0.8:", ((C-np.eye(len(X))).max(1)>0.8).sum())
# spectral clustering / kmeans with k=28 on unit vectors
from sklearn.cluster import KMeans
for k in [20,28,40]:
    km = KMeans(n_clusters=k,n_init=5,random_state=0).fit(X)
    cent = km.cluster_centers_/np.linalg.norm(km.cluster_centers_,axis=1,keepdims=True)
    sizes = np.bincount(km.labels_,minlength=k)
    within = np.array([ (X[km.labels_==c]@cent[c]).mean() for c in range(k)])
    print(f"k={k}: sizes {np.sort(sizes)[::-1]}  mean within-cos {within.mean():.2f}  centroid pairwise max cos {np.sort((cent@cent.T-np.eye(k)).max(1))[::-1][:5]}")
```