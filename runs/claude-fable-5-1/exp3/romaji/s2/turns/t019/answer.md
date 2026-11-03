```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
from sklearn.cluster import AgglomerativeClustering
d=384; hd=64
def blk(nm,l,j):
    s=slice(j*hd,(j+1)*hd); return W[f'{nm}.{l}'][s]
def rowspace(A,k):
    U,S,Vt=np.linalg.svd(A,full_matrices=False); return Vt[:k].T
Q3=rowspace(np.vstack([blk('W_q',0,3),blk('W_k',0,3)]),30)
Pq=np.eye(d)-Q3@Q3.T
fc=W['W_fc.0']; rn=np.linalg.norm(fc,axis=1); A=fc[rn>1]
# project out position subspace, keep rest
B=A@Pq; Bn=B/np.linalg.norm(B,axis=1,keepdims=True)
C=Bn@Bn.T
np.fill_diagonal(C,0)
print("max cos per neuron percentiles:", np.percentile(C.max(1),[5,25,50,75,95]))
# count pairs with high cosine
print("fraction of pairs with cos>0.8:", (C>0.8).sum()/C.size, ">0.6:", (C>0.6).sum()/C.size)
for thr in [0.5,0.6,0.7,0.8]:
    cl=AgglomerativeClustering(n_clusters=None,distance_threshold=1-thr,metric='cosine',linkage='average').fit(Bn)
    labels=cl.labels_; sizes=np.bincount(labels)
    print(f"thr {thr}: n clusters {labels.max()+1}, clusters with size>=5: {(sizes>=5).sum()}, sizes top: {np.sort(sizes)[::-1][:30]}")
# also check: are neuron read directions roughly within a low-dim subspace? spectrum of Bn
S=np.linalg.svd(Bn,compute_uv=False); print("sv of normalized rows:", S[:40])
```