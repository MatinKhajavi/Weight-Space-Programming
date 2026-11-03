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
Aa=A[alive]
coef=np.linalg.lstsq(E.T,Aa.T,rcond=None)[0].T   # 525 x 25
coefn=coef/an[alive][:,None]
C=np.corrcoef(coefn.T)
import pandas as pd; pd.set_option('display.width',300); pd.set_option('display.max_columns',40)
print("correlation of token coefficients across MLP0 neurons:")
print(pd.DataFrame(C).round(1).to_string())
# pairs with corr > 0.3
print("pairs corr>0.3:", [(i,j,round(float(C[i,j]),2)) for i in range(25) for j in range(i+1,25) if C[i,j]>0.3])
print("pairs corr<-0.3:", [(i,j,round(float(C[i,j]),2)) for i in range(25) for j in range(i+1,25) if C[i,j]<-0.3])
# neurons reading many tokens positively: distribution of number of tokens with coef > 0.25
cnt=(coefn>0.25).sum(1); print("num tokens read positively per neuron hist:", np.bincount(cnt))
# for neurons reading >=3 tokens positively, which tokens?
multi=np.where(cnt>=3)[0]
print("token frequency among multi-token neurons:", np.bincount(np.where(coefn[multi]>0.25)[1],minlength=25))
```