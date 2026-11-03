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
def QK(l,j):
    sl=slice(j*hd,(j+1)*hd); lw=W[f'ln1_w.{l}']; return (W[f'W_q.{l}'][sl]*lw).T@(W[f'W_k.{l}'][sl]*lw)
# reference logit directions from L3H1+L3H5 and separately from L2H1+L3H0
def refU(heads):
    M=sum(OV(l,j) for l,j in heads); Ui=(U.T@M@E21.T).T; return Ui/np.linalg.norm(Ui,axis=1)[:,None]
UnA=refU([(3,1),(3,5)]); UnB=refU([(2,1),(3,0)])
print("copy fidelity: diag mean / offdiag abs mean of Un @ U^T OV_h E (using ref A for L2/L3H0.. and ref B for L3H1/5)")
for l in range(4):
    for j in range(6):
        Ui=(U.T@OV(l,j)@E21.T).T
        ref = UnB if (l,j) in [(3,1),(3,5)] else UnA
        D=ref@Ui.T   # 21x21: ref token y . image of x
        d=np.diag(D); off=D-np.diag(d)
        print(f"L{l}H{j}: diag mean {d.mean():.2f} (min {d.min():.2f}), offdiag mean {off.mean():.3f}, offdiag abs mean {np.abs(off).mean():.2f}, ratio {d.mean()/np.abs(off).mean():.1f}")
# induction QK signature in token basis: q current token (E) vs key prev token (I5, I1)
I5=(OV0[5]@E21.T).T; I1=(OV0[1]@E21.T).T
def on(M): return M/np.linalg.norm(M,axis=1)[:,None]
print("QK blocks: diag-mean of E^T QK I5 (cur token i matches key's prev token = induction), of E^T QK E (same token), I5^T QK I5")
for l in range(1,4):
    for j in range(6):
        M=QK(l,j)
        b1=E21@M@on(I5).T; b2=E21@M@E21.T; b3=on(I5)@M@on(I5).T; b4=E21@M@on(I1).T
        def stat(Bm):
            d=np.diag(Bm); off=Bm-np.diag(d); return f"{d.mean():.3f}/{np.abs(off).mean():.3f}"
        print(f"L{l}H{j}: E-I5 {stat(b1)} | E-E {stat(b2)} | I5-I5 {stat(b3)} | E-I1 {stat(b4)}")
```