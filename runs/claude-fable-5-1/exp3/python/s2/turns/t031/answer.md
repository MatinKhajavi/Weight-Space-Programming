```python
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
lw0=W['ln1_w.0']; L0={}
for j in range(6):
    sl=slice(j*hd,(j+1)*hd)
    for nm in 'qkv': L0[f'{j}{nm}']=W[f'W_{nm}.0'][sl]*lw0[None,:]
Pm=np.vstack([L0['1q'],L0['1k'],L0['5q'],L0['5k']]); P=np.linalg.svd(Pm,full_matrices=False)[2][:24].T
outs=[]; OV0={}
for h in range(6):
    sl=slice(h*hd,(h+1)*hd); OV=W['W_o.0'][:,sl]@L0[f'{h}v']; OV0[h]=OV
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
E=np.array([Xn[labs==c].mean(0) for c in order]); E/=np.linalg.norm(E,axis=1)[:,None]
Aa=A[alive]; An=Aa/an[alive][:,None]
print("token: #pure, #neurons cos>0.3 with e_x, with H5-image, with H1-image, with H2-image; norms of images")
for x in range(25):
    e=E[x]
    im5=OV0[5]@e; im1=OV0[1]@e; im2=OV0[2]@e; im3=OV0[3]@e
    c0=An@e; c5=An@(im5/np.linalg.norm(im5)); c1=An@(im1/np.linalg.norm(im1)); c2=An@(im2/np.linalg.norm(im2))
    print(x, sizes[order][x], (c0>0.3).sum(), (c5>0.3).sum(), (c1>0.3).sum(), (c2>0.3).sum(), " |H5 e|=%.2f |H1 e|=%.2f |H2 e|=%.2f |H3 e|=%.2f" % (np.linalg.norm(im5),np.linalg.norm(im1),np.linalg.norm(im2),np.linalg.norm(im3)))
# L0H5 key / L0H1 key along token dirs (does the prev-token head avoid/prefer certain tokens as keys)
for h in [1,5,2,3,0]:
    K=L0[f'{h}k']; Q=L0[f'{h}q']
    print(f"L0H{h} ||K e_x||:", np.round(np.linalg.norm(K@E.T,axis=0),2))
```