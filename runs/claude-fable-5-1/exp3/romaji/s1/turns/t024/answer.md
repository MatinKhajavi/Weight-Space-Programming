```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
Pj=W['W_proj.3']   # 384 x 1536
Up,Sp,_=np.linalg.svd(Pj); L=Up[:,:27]
norms=np.linalg.norm(Pj,axis=0)
print("neuron output norms: percentiles", np.percentile(norms,[10,25,50,75,90,99]))
strong=np.where(norms>np.percentile(norms,50))[0]
X=Pj[:,strong]; X=X/np.linalg.norm(X,axis=0)
C=X.T@X
iu=np.triu_indices(len(strong),1)
vals=np.abs(C[iu])
print("pairwise |cos| among strong neurons: percentiles", np.percentile(vals,[50,90,99,99.9]), "frac>0.8:",(vals>0.8).mean(), "frac>0.9:",(vals>0.9).mean())
# Also the fraction of each neuron's output in L
fr=np.linalg.norm(L.T@Pj,axis=0)**2/norms**2
print("frac of neuron output in L (27-d): percentiles", np.percentile(fr,[10,25,50,75,90]))
print("weighted by norm^2: ", (fr*norms**2).sum()/(norms**2).sum())
# how many distinct directions: greedy clustering with |cos|>0.85
Y=X.copy()
order=np.argsort(-norms[strong])
assigned=-np.ones(len(strong),int); centers=[]
for idx in order:
    if assigned[idx]>=0: continue
    v=Y[:,idx]
    c=Y.T@v
    members=np.where((np.abs(c)>0.85)&(assigned<0))[0]
    assigned[members]=len(centers)
    centers.append((idx,len(members)))
print("number of greedy clusters:",len(centers))
print("cluster sizes (top 40):",sorted([m for _,m in centers],reverse=True)[:40])
```