```python
np.set_printoptions(precision=3, suppress=True, linewidth=250)
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
E=np.array([Xn[labs==c].mean(0) for c in order]); E/=np.linalg.norm(E,axis=1)[:,None]; E21=E[:21]
def QK(l,j):
    sl=slice(j*hd,(j+1)*hd); lw=W[f'ln1_w.{l}']; return (W[f'W_q.{l}'][sl]*lw).T@(W[f'W_k.{l}'][sl]*lw)
def on(M): return M/np.linalg.norm(M,axis=1)[:,None]
I5=on((OV0[5]@E21.T).T); I1=on((OV0[1]@E21.T).T); I2=on((OV0[2]@E21.T).T); I3=on((OV0[3]@E21.T).T)
def stat(Bm):
    d=np.diag(Bm); off=Bm-np.diag(d); return f"{d.mean():+.3f}/{np.abs(off).mean():.3f}"
print("head: I5-I1 | I1-I5 | I1-I1 | E-I1 | I1-E | I5-E | E-I2 | I2-I2")
for l,j in [(2,1),(3,0),(3,1),(3,5),(2,2),(1,1),(1,3)]:
    M=QK(l,j)
    print(f"L{l}H{j}: {stat(I5@M@I1.T)} | {stat(I1@M@I5.T)} | {stat(I1@M@I1.T)} | {stat(E21@M@I1.T)} | {stat(I1@M@E21.T)} | {stat(I5@M@E21.T)} | {stat(E21@M@I2.T)} | {stat(I2@M@I2.T)}")
# Also: do the MLP0 / MLP1 neurons read I1 and I5 of the same token jointly (which would be odd if both were offset 1: redundant) - correlation of coefficients
D=np.vstack([E21,I5,I1]); Aa=A[alive]; coef=np.linalg.lstsq(D.T,Aa.T,rcond=None)[0].T; coefn=coef/an[alive][:,None]
c5=coefn[:,21:42]; c1=coefn[:,42:63]
print("per-token corr between I5-coef and I1-coef across MLP0 neurons:", np.round([np.corrcoef(c5[:,x],c1[:,x])[0,1] for x in range(21)],2))
A1=W['W_fc.1']*W['ln2_w.1'][None,:]; coef1=np.linalg.lstsq(D.T,A1.T,rcond=None)[0].T; coef1n=coef1/np.linalg.norm(A1,axis=1)[:,None]
c5b=coef1n[:,21:42]; c1b=coef1n[:,42:63]
print("MLP1 per-token corr I5 vs I1 coef:", np.round([np.corrcoef(c5b[:,x],c1b[:,x])[0,1] for x in range(21)],2))
print("MLP1: #neurons |coef|>0.25 on I5:", (np.abs(c5b).max(1)>0.25).sum(), " on I1:", (np.abs(c1b).max(1)>0.25).sum(), " on E:", (np.abs(coef1n[:,:21]).max(1)>0.25).sum())
```