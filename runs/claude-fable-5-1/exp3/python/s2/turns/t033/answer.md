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
E21=E[:21]
Aa=A[alive]
# regress on H5 images and H1 images jointly with token dirs
I5=(OV0[5]@E21.T).T; I1=(OV0[1]@E21.T).T
print("H5 image gram offdiag abs mean:", np.abs(np.corrcoef(I5)-np.eye(21)).mean(), " H1:", np.abs(np.corrcoef(I1)-np.eye(21)).mean())
D=np.vstack([E21,I5/np.linalg.norm(I5,axis=1)[:,None],I1/np.linalg.norm(I1,axis=1)[:,None]])   # 63 x 384
coef=np.linalg.lstsq(D.T,Aa.T,rcond=None)[0].T; coefn=coef/an[alive][:,None]
print("explained energy fraction:", 1-np.linalg.norm(Aa-coef@D)**2/np.linalg.norm(Aa)**2)
c5=coefn[:,21:42]; c1=coefn[:,42:63]; c0=coefn[:,:21]
print("neurons with max |coef| on H5-images > 0.3:", (np.abs(c5).max(1)>0.3).sum(), " on H1-images:", (np.abs(c1).max(1)>0.3).sum(), " on tokens:", (np.abs(c0).max(1)>0.3).sum())
import pandas as pd; pd.set_option('display.width',300); pd.set_option('display.max_columns',40)
C5=np.corrcoef(c5.T); 
print("corr of H5-image (prev-token) coefficients:"); print(pd.DataFrame(C5).round(1).to_string())
print("pairs corr>0.3:", [(i,j,round(float(C5[i,j]),2)) for i in range(21) for j in range(i+1,21) if C5[i,j]>0.3])
# eigen-structure of C5: a block of co-read tokens shows as a dominant eigenvector with many same-sign entries
w,v=np.linalg.eigh(C5); print("top eigvals:", w[-4:][::-1]); print("top eigvec:", np.round(v[:,-1],2)); print("2nd eigvec:", np.round(v[:,-2],2))
C1=np.corrcoef(c1.T); w1,v1=np.linalg.eigh(C1); print("H1 top eigvals:", w1[-4:][::-1]); print("H1 top eigvec:", np.round(v1[:,-1],2))
# cross: do neurons reading current token x also read prev-token y? -> bigram detectors; summarize count of bigram-detector neurons
bg=((np.abs(c0).max(1)>0.3)&(np.abs(c5).max(1)>0.3)).sum(); print("neurons reading both a current token and an H5 prev-token strongly:", bg)
```