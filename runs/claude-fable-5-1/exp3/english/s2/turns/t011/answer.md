```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
from scipy.cluster.hierarchy import linkage, fcluster
d=384; hd=64; H=6
g=W["ln1_w.0"]; g2=W["ln2_w.0"]
fc=W["W_fc.0"]; rn=np.linalg.norm(fc,axis=1); strong=rn>0.8
fce=(fc*g2)[strong]
Uf,Sf,Vf=np.linalg.svd(fce,full_matrices=False)
for k in [20,30,40]:
    T=Vf[:k].T
    X=fce@T; Xn=X/np.linalg.norm(X,axis=1,keepdims=True)
    C=Xn@Xn.T
    Z=linkage(Xn,method="average",metric="cosine")
    for th in [0.3,0.5]:
        lab=fcluster(Z,t=th,criterion="distance")
        sizes=np.sort(np.bincount(lab)[1:])[::-1]
        print(f"k={k} th={th}: nclusters {lab.max()}, sizes {sizes[:40]}")
# look at distribution of pairwise cosines
T=Vf[:30].T; X=fce@T; Xn=X/np.linalg.norm(X,axis=1,keepdims=True); C=Xn@Xn.T
iu=np.triu_indices(len(Xn),1)
print("pairwise cosine quantiles:", np.quantile(C[iu],[.5,.9,.99,.999]), " count >0.8:", (C[iu]>0.8).sum(), " >0.9:", (C[iu]>0.9).sum())
# how 'one-hot' are neurons: for each neuron, the max |cos| with any other neuron
print("per-neuron max cos with others quantiles:", np.quantile((C-np.eye(len(C))).max(1),[.1,.25,.5,.75,.9]))
```