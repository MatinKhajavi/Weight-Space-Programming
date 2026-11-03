```python
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
def setup():
    lw0=W['ln1_w.0']; L0={}
    for j in range(6):
        sl=slice(j*hd,(j+1)*hd)
        for nm in 'qkv': L0[f'{j}{nm}']=W[f'W_{nm}.0'][sl]*lw0[None,:]
    Pm=np.vstack([L0['1q'],L0['1k'],L0['5q'],L0['5k']]); P=np.linalg.svd(Pm,full_matrices=False)[2][:24].T
    outs=[]
    for h in range(6):
        sl=slice(h*hd,(h+1)*hd); OV=W['W_o.0'][:,sl]@L0[f'{h}v']
        u,s,vt=np.linalg.svd(OV,full_matrices=False); outs.append(u[:,:{0:3,4:3,1:27,5:27,2:10,3:10}[h]])
    S=np.hstack([P]+outs); Qs,_=np.linalg.qr(S)
    def clean(M):
        R=M-(M@Qs)@Qs.T; return R/np.linalg.norm(R)
    A=W['W_fc.0']*W['ln2_w.0'][None,:]; an=np.linalg.norm(A,axis=1); Aa=A[an>0.8]
    blocks=[clean(Aa)*3]
    for h in [1,5,2,3]: blocks.append(clean(L0[f'{h}v']))
    lw=W['ln1_w.1']
    for j in range(6):
        sl=slice(j*hd,(j+1)*hd)
        blocks.append(clean(W['W_q.1'][sl]*lw)); blocks.append(clean(W['W_v.1'][sl]*lw))
    A1=W['W_fc.1']*W['ln2_w.1'][None,:]; blocks.append(clean(A1)*2)
    X=np.vstack(blocks); u,s,vt=np.linalg.svd(X,full_matrices=False); Tok=vt[:28].T
    u,s,vt=np.linalg.svd(W['W_proj.3'],full_matrices=False); U=u[:,:28]
    return L0,P,outs,Tok,U
L0,P,outs,Tok,U=setup()
A=W['W_fc.0']*W['ln2_w.0'][None,:]; an=np.linalg.norm(A,axis=1); alive=np.where(an>0.8)[0]; Aa=A[alive]
Z=Aa@Tok; zn=np.linalg.norm(Z,axis=1); Zn=Z/zn[:,None]
from sklearn.cluster import AgglomerativeClustering
cl=AgglomerativeClustering(n_clusters=None,distance_threshold=0.3,metric='cosine',linkage='average').fit(Zn)
labs=cl.labels_; sizes=np.bincount(labs); order=np.argsort(-sizes)
big=[c for c in order if sizes[c]>=3]
cent=np.array([Zn[labs==c].mean(0) for c in big]); cent/=np.linalg.norm(cent,axis=1)[:,None]
CC=cent@cent.T
print("n clusters>=3:", len(big), "sizes:", sizes[big])
print("pairs of cluster centres with cos>0.4:", [(i,j,round(float(CC[i,j]),2)) for i in range(len(big)) for j in range(i+1,len(big)) if CC[i,j]>0.4])
print("pairs with cos<-0.4:", [(i,j,round(float(CC[i,j]),2)) for i in range(len(big)) for j in range(i+1,len(big)) if CC[i,j]<-0.4])
# distribution of off-diagonal centre cosines
iu=np.triu_indices(len(big),1); print("centre cos histogram:", np.histogram(CC[iu],bins=np.linspace(-1,1,11))[0])
B=W['W_proj.0'][:,alive]
outU=np.array([(U.T@B[:,labs==c]).mean(1) for c in big]); outUn=outU/np.linalg.norm(outU,axis=1)[:,None]
OO=outUn@outUn.T
print("output-direction cosines between clusters (rows = clusters in size order):")
import pandas as pd; pd.set_option('display.width',300); pd.set_option('display.max_columns',50)
print(pd.DataFrame(OO).round(1).to_string())
```