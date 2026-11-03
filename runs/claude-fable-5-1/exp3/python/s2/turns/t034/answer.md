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
u,s,vt=np.linalg.svd(W['W_proj.3'],full_matrices=False); U=u[:,:28]
def OV(l,j):
    sl=slice(j*hd,(j+1)*hd); return W[f'W_o.{l}'][:,sl]@(W[f'W_v.{l}'][sl]*W[f'ln1_w.{l}'][None,:])
Csum=OV(3,1)+OV(3,5)+OV(2,1)+OV(3,0)
Uimg=(U.T@Csum@E21.T).T; Un=Uimg/np.linalg.norm(Uimg,axis=1)[:,None]
Aa=A[alive]; coef=np.linalg.lstsq(E21.T,Aa.T,rcond=None)[0].T; coefn=coef/an[alive][:,None]
B=W['W_proj.0'][:,alive]; OUT=(U.T@B).T@Un.T    # nalive x 21 : decoded logit contribution of each neuron
# weight neurons by positive token coefficient
Wt=np.clip(coefn,0,None); Wt[Wt<0.3]=0
Mf=Wt.T@OUT   # 21 x 21
import pandas as pd; pd.set_option('display.width',300); pd.set_option('display.max_columns',40)
print("M_full:"); print(pd.DataFrame(Mf).round(1).to_string())
print("diag:", np.round(np.diag(Mf),2))
Mp=np.clip(Mf,0,None); Mpn=Mp/ (np.linalg.norm(Mp,axis=1)[:,None]+1e-9)
S=Mpn@Mpn.T
print("row similarity of positive parts:"); print(pd.DataFrame(S).round(1).to_string())
print("column sums of positive part:", np.round(Mp.sum(0),1))
print("row sums of positive part:", np.round(Mp.sum(1),1))
# cluster rows
cl2=AgglomerativeClustering(n_clusters=None,distance_threshold=0.5,metric='cosine',linkage='average').fit(Mpn)
print("row clusters:", cl2.labels_)
```