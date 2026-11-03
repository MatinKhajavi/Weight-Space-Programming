```python
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
lw0=W['ln1_w.0']; L0={}
for j in range(6):
    sl=slice(j*hd,(j+1)*hd)
    for nm in 'qkv': L0[f'{j}{nm}']=W[f'W_{nm}.0'][sl]*lw0[None,:]
Pm=np.vstack([L0['1q'],L0['1k'],L0['5q'],L0['5k']]); P=np.linalg.svd(Pm,full_matrices=False)[2][:24].T
outs=[]
for h in range(6):
    sl=slice(h*hd,(h+1)*hd); OV=W['W_o.0'][:,sl]@L0[f'{h}v']
    u,s,vt=np.linalg.svd(OV,full_matrices=False); outs.append(u[:,:{0:3,4:3,1:27,5:27,2:10,3:10}[h]])
Tm=np.vstack([L0['1v'],L0['5v']]); T=np.linalg.svd(Tm,full_matrices=False)[2][:30].T
A=W['W_fc.0']*W['ln2_w.0'][None,:]; an=np.linalg.norm(A,axis=1); alive=np.where(an>0.8)[0]
def fr(M,S): return np.linalg.norm(M@S)**2/np.linalg.norm(M)**2
Hall=np.hstack(outs)
fT=np.array([fr(A[i:i+1],T) for i in alive]); fH=np.array([fr(A[i:i+1],Hall) for i in alive]); fP=np.array([fr(A[i:i+1],P) for i in alive])
print("alive:", len(alive), " T-frac hist:", np.histogram(fT,bins=np.linspace(0,1,11))[0])
print("headout-frac hist:", np.histogram(fH,bins=np.linspace(0,1,11))[0])
pure=alive[(fT>0.4)&(fH<0.25)]
print("pure token readers:", len(pure))
X=A[pure]; Xn=X/np.linalg.norm(X,axis=1)[:,None]
C=Xn@Xn.T; iu=np.triu_indices(len(pure),1)
print("cos hist among pure readers:", np.histogram(C[iu],bins=np.linspace(-1,1,21))[0])
from sklearn.cluster import AgglomerativeClustering
for thr in [0.6,0.7,0.8,0.9]:
    cl=AgglomerativeClustering(n_clusters=None,distance_threshold=1-thr,metric='cosine',linkage='average').fit(Xn)
    sizes=np.bincount(cl.labels_); print(f"thr {thr}: nclusters {len(sizes)} sizes {np.sort(sizes)[::-1]}")
cl=AgglomerativeClustering(n_clusters=None,distance_threshold=0.3,metric='cosine',linkage='average').fit(Xn)
labs=cl.labels_; sizes=np.bincount(labs); order=np.argsort(-sizes)
cent=np.array([Xn[labs==c].mean(0) for c in order]); cent/=np.linalg.norm(cent,axis=1)[:,None]
CC=cent@cent.T
print("centre cosine matrix (clusters in size order):")
print(np.round(CC,1))
```