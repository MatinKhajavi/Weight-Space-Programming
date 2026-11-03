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
E=np.array([Xn[labs==c].mean(0) for c in order]); E/=np.linalg.norm(E,axis=1)[:,None]
print("Gram of token dirs (abs>0.3 pairs):", [(i,j,round(float(E[i]@E[j]),2)) for i in range(25) for j in range(i+1,25) if abs(E[i]@E[j])>0.3])
# orthonormalize E (25) and remove along with P, head outs from all alive neurons
Qe,_=np.linalg.qr(E.T)
S=np.hstack([P,Hall,Qe]); Qs,_=np.linalg.qr(S)
Aa=A[alive]
R=Aa-(Aa@Qs)@Qs.T
print("energy fraction remaining after removing P, head outs, 25 token dirs:", np.linalg.norm(R)**2/np.linalg.norm(Aa)**2)
u,s,vt=np.linalg.svd(R,full_matrices=False)
print("residual sv:", s[:30])
# how many neurons have strong residual direction? examine top residual directions: who reads them
for i in range(6):
    v=vt[i]; c=(Aa@v)/an[alive]
    print(f"resid dir {i}: sv {s[i]:.2f}; n neurons with |cos|>0.4: {(np.abs(c)>0.4).sum()}, >0.6: {(np.abs(c)>0.6).sum()}; energy in T {np.linalg.norm(v@T)**2:.2f}; in W_proj.0 top100 colspace {np.sum((np.linalg.svd(W['W_proj.0'],full_matrices=False)[0][:,:100].T@v)**2):.2f}")
# fraction of each alive neuron's energy explained by E (token dirs)
fe=np.array([fr(A[i:i+1],Qe) for i in alive]); print("neuron energy in 25 token dirs: hist", np.histogram(fe,bins=np.linspace(0,1,11))[0])
```