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
# token clusters -> neuron index lists with signs
clusters=[]
for c in range(1,lab.max()+1):
    idx=np.where(lab==c)[0]
    if len(idx)<2: continue
    V=X[idx]; sg=np.sign(V@V[0])
    clusters.append((sel[idx], sg))
print("n clusters", len(clusters))
# W_proj.0 SVD
Pj = W["W_proj.0"]   # 384 x 1536
uu,ss,vv = np.linalg.svd(Pj,full_matrices=False)
print("W_proj.0 sv:", ss[:15])
# loadings of neurons on top components (weighted by sign of neuron's reading of its token dir)
for comp in range(4):
    load = vv[comp]*ss[comp]
    row=[]
    for (idx,sg) in clusters:
        row.append((load[idx]*sg).mean())
    row=np.array(row)
    print(f"comp {comp}: per-token-cluster mean loading:", row.round(2), " | #pos %d #neg %d" % ((row>0.05).sum(), (row<-0.05).sum()))
# Better: output directions of token-cluster neurons: for each cluster, mean output vector (W_proj.0 columns) with sign; then pairwise cosine matrix among clusters
outs=[]
for (idx,sg) in clusters:
    o = (Pj[:,idx]*sg).mean(1); outs.append(o)
outs=np.array(outs); on = outs/np.linalg.norm(outs,axis=1,keepdims=True)
Cm = on@on.T
print("cluster output-direction cosine matrix: eigenvalues of Cm:", np.linalg.eigvalsh(Cm)[::-1][:6].round(2))
# cluster the token clusters by output similarity into 2 groups
ev,evec=np.linalg.eigh(Cm)
g = evec[:,-1]>0
print("group split by top eigvec:", g.sum(), (~g).sum(), " mean within cos A %.2f B %.2f between %.2f" % (Cm[g][:,g].mean(), Cm[~g][:,~g].mean(), Cm[g][:,~g].mean()))
# sorted cosine matrix print
order=np.argsort(evec[:,-1])
print((Cm[order][:,order]).round(1))
```