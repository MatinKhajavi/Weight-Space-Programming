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
u,s,vt=np.linalg.svd(W['W_proj.3'],full_matrices=False); U=u[:,:28]
def OV(l,j):
    sl=slice(j*hd,(j+1)*hd); return W[f'W_o.{l}'][:,sl]@(W[f'W_v.{l}'][sl]*W[f'ln1_w.{l}'][None,:])
Csum=OV(3,1)+OV(3,5)+OV(2,1)+OV(3,0)
Uimg=(U.T@Csum@E.T).T; Un=Uimg/np.linalg.norm(Uimg,axis=1)[:,None]   # 25 x 28 unit logit dirs
# MLP0 outputs of each cluster
B=W['W_proj.0']
O=np.array([ (B[:,pure[labs==c]]).sum(1) for c in order ])   # 25 x 384
OU=O@U   # 25 x 28
M=OU@Un.T   # 25 x 25 : token x -> logit contribution to token y
print("cluster sizes:", sizes[order])
print("MLP0 output norm per cluster:", np.round(np.linalg.norm(O,axis=1),2), " fraction in U:", np.round(np.linalg.norm(OU,axis=1)**2/np.linalg.norm(O,axis=1)**2,2))
import pandas as pd; pd.set_option('display.width',300); pd.set_option('display.max_columns',40)
print("Bigram-like matrix M[x,y] (row = current token cluster, col = predicted token):")
print(pd.DataFrame(M).round(1).to_string())
print("diagonal:", np.round(np.diag(M),2))
# row-normalized correlation structure: which rows are similar
Mn=M/np.linalg.norm(M,axis=1)[:,None]
print("row-similarity (cos) matrix:")
print(pd.DataFrame(Mn@Mn.T).round(1).to_string())
```