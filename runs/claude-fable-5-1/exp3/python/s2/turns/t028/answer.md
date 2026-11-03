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
fT=np.array([fr(A[i:i+1],T) for i in alive]); fH=np.array([fr(A[i:i+1],Hall) for i in alive])
pure=alive[(fT>0.4)&(fH<0.25)]
X=A[pure]; Xn=X/np.linalg.norm(X,axis=1)[:,None]
from sklearn.cluster import AgglomerativeClustering
cl=AgglomerativeClustering(n_clusters=None,distance_threshold=0.3,metric='cosine',linkage='average').fit(Xn)
labs=cl.labels_; sizes=np.bincount(labs); order=np.argsort(-sizes)
E=np.array([Xn[labs==c].mean(0) for c in order]); E/=np.linalg.norm(E,axis=1)[:,None]   # token directions (25 x 384)
print("token dirs: energy in P:", [round(float(np.linalg.norm(e@P)**2),2) for e in E])
print("token dirs: energy in T:", [round(float(np.linalg.norm(e@T)**2),2) for e in E])
# unembedding subspace and copy map
u,s,vt=np.linalg.svd(W['W_proj.3'],full_matrices=False); U=u[:,:28]
def OV(l,j):
    sl=slice(j*hd,(j+1)*hd); return W[f'W_o.{l}'][:,sl]@(W[f'W_v.{l}'][sl]*W[f'ln1_w.{l}'][None,:])
for name,M in [('L3H1',OV(3,1)),('L3H5',OV(3,5)),('L2H1',OV(2,1)),('L3H0',OV(3,0)),('sum',OV(3,1)+OV(3,5)+OV(2,1)+OV(3,0))]:
    Ux=(U.T@M@E.T).T   # 25 x 28 : image of each token dir in U
    Uxn=Ux/np.linalg.norm(Ux,axis=1)[:,None]
    G=Uxn@Uxn.T; iu=np.triu_indices(len(E),1)
    print(f"{name}: norms of images:", np.round(np.linalg.norm(Ux,axis=1),2))
    print(f"   off-diag cos of images: mean {G[iu].mean():.2f}, hist", np.histogram(G[iu],bins=np.linspace(-1,1,11))[0])
```