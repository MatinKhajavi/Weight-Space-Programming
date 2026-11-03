```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
Tok=np.load('/tmp/Tok.npy')
A=W['W_fc.0']*W['ln2_w.0'][None,:]; an=np.linalg.norm(A,axis=1); alive=np.where(an>0.8)[0]; Aa=A[alive]
Z=Aa@Tok   # 525 x 28
zn=np.linalg.norm(Z,axis=1)
print("fraction of neuron energy in Tok: percentiles", np.percentile(zn**2/an[alive]**2,[10,25,50,75,90]))
Zn=Z/zn[:,None]
Cos=Zn@Zn.T
iu=np.triu_indices(len(alive),1)
print("histogram of pairwise cosines:", np.histogram(Cos[iu],bins=np.linspace(-1,1,21)))
# cluster with agglomerative clustering on cosine
from sklearn.cluster import AgglomerativeClustering
for thr in [0.5,0.7]:
    cl=AgglomerativeClustering(n_clusters=None,distance_threshold=1-thr,metric='cosine',linkage='average').fit(Zn)
    labs=cl.labels_; sizes=np.bincount(labs)
    print(f"thr {thr}: n clusters {len(sizes)}; sizes sorted:", np.sort(sizes)[::-1][:40])
```