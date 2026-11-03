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
u,s,vt=np.linalg.svd(W['W_proj.3'],full_matrices=False); U=u[:,:28]
def OV(l,j):
    sl=slice(j*hd,(j+1)*hd); return W[f'W_o.{l}'][:,sl]@(W[f'W_v.{l}'][sl]*W[f'ln1_w.{l}'][None,:])
Csum=OV(3,1)+OV(3,5)+OV(2,1)+OV(3,0)
Uimg=(U.T@Csum@E21.T).T; Un=Uimg/np.linalg.norm(Uimg,axis=1)[:,None]
I5=(OV0[5]@E21.T).T; I5n=I5/np.linalg.norm(I5,axis=1)[:,None]
I1=(OV0[1]@E21.T).T; I1n=I1/np.linalg.norm(I1,axis=1)[:,None]
D=np.vstack([E21,I5n,I1n])
Aa=A[alive]; coef=np.linalg.lstsq(D.T,Aa.T,rcond=None)[0].T; coefn=coef/an[alive][:,None]
B=W['W_proj.0'][:,alive]; OUT=(U.T@B).T@Un.T   # decoded outputs
c0=coefn[:,:21]; c5=coefn[:,21:42]; c1=coefn[:,42:63]
# neurons with cur=space(4)
sel=np.where(c0[:,4]>0.3)[0]
print("neurons reading cur=token4:", len(sel))
print("  idx | coef cur4 | coef prev(H5)=4 | prev(H5)=0 | prev(H1)=4 | out->4 | out->0 | top out tokens")
for n in sel[np.argsort(-c0[sel,4])][:30]:
    top=np.argsort(-OUT[n])[:3]
    print(f"  {alive[n]:5d} | {c0[n,4]:.2f} | {c5[n,4]:.2f} | {c5[n,0]:.2f} | {c1[n,4]:.2f} | {OUT[n,4]:.2f} | {OUT[n,0]:.2f} | {top} {np.round(OUT[n,top],2)}")
# summary: neurons with cur=4 and prev5=4 (>0.2): their mean out->4
g=sel[c5[sel,4]>0.2]; print("cur=4 & prev=4 neurons:", len(g), " mean out->4:", OUT[g,4].mean() if len(g) else None, " mean out->0:", OUT[g,0].mean() if len(g) else None)
g2=sel[c5[sel,0]>0.2]; print("cur=4 & prev=0(newline) neurons:", len(g2), " mean out->4:", OUT[g2,4].mean() if len(g2) else None)
g3=sel[(c5[sel,4]<0.1)&(c5[sel,0]<0.1)]; print("cur=4 & no prev reading:", len(g3), " mean out->4:", OUT[g3,4].mean(), " mean out->0:", OUT[g3,0].mean())
# same analysis for cur=newline (0)
sel0=np.where(c0[:,0]>0.3)[0]; print("cur=0 neurons:", len(sel0), " mean out->4:", OUT[sel0,4].mean(), " mean out->0:", OUT[sel0,0].mean(), " top mean out tokens:", np.argsort(-OUT[sel0].mean(0))[:5], np.round(np.sort(OUT[sel0].mean(0))[::-1][:5],2))
# for other tokens: mean out->same token (self) for pure cur readers
for x in range(21):
    s_=np.where((c0[:,x]>0.3)&(np.abs(c5).max(1)<0.15))[0]
    if len(s_): print(f"token {x}: n={len(s_)} mean self-out {OUT[s_,x].mean():.2f}  mean out->0 {OUT[s_,0].mean():.2f} mean out->4 {OUT[s_,4].mean():.2f}")
```