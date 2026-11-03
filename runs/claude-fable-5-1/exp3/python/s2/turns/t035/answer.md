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
E=np.array([Xn[labs==c].mean(0) for c in order]); E/=np.linalg.norm(E,axis=1)[:,None]; E21=E[:21]
Aa=A[alive]; coef=np.linalg.lstsq(E21.T,Aa.T,rcond=None)[0].T; coefn=coef/an[alive][:,None]
B=W['W_proj.0'][:,alive]
# t* direction (layer-1 shared key direction, residual)
lw=W['ln1_w.1']; Ks=np.vstack([W['W_k.1'][j*hd:(j+1)*hd]*lw[None,:] for j in range(6)])
Qs,_=np.linalg.qr(np.hstack([P,outs[0],outs[4]])); Kr=Ks-(Ks@Qs)@Qs.T
u,s,vt=np.linalg.svd(Kr,full_matrices=False); tstar=vt[0]
Wt=np.clip(coefn,0,None); Wt[Wt<0.3]=0
drive=Wt.T@(B.T@tstar)   # per token: weighted sum of outputs along t*
print("t* drive per token (weighted by positive token coefficient):", np.round(drive,2))
print("direct linear drive t*^T W_proj0 W_fc0 e_x:", np.round(tstar@W['W_proj.0']@A@E21.T,2))
# Position summary (H0o,H4o) coupling: which tokens' neurons also read H0o/H4o
pos=np.hstack([outs[0],outs[4]])
fpos=np.array([fr(A[i:i+1],pos) for i in alive])
for x in [0,4,3,1,2]:
    sel=coefn[:,x]>0.3
    print(f"token {x}: {sel.sum()} neurons; mean pos-summary energy {fpos[sel].mean():.3f} (overall {fpos.mean():.3f}); mean H5o energy {np.mean([fr(A[i:i+1],outs[5]) for i in alive[sel]]):.3f}; mean H2o energy {np.mean([fr(A[i:i+1],outs[2]) for i in alive[sel]]):.3f}")
# Which tokens do the L1 heads' V read (for copying), and queries?
for j in range(6):
    sl=slice(j*hd,(j+1)*hd)
    V=W['W_v.1'][sl]*lw[None,:]; Q=W['W_q.1'][sl]*lw[None,:]; K=W['W_k.1'][sl]*lw[None,:]
    print(f"L1H{j} ||V e_x||:", np.round(np.linalg.norm(V@E21.T,axis=0),2)[:8], " ||K e_x||:", np.round(np.linalg.norm(K@E21.T,axis=0),2)[:8], " ||Q e_x||:", np.round(np.linalg.norm(Q@E21.T,axis=0),2)[:8])
```