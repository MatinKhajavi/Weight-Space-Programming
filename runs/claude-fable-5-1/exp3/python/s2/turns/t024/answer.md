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
# t* from layer-1 keys
lw=W['ln1_w.1']
Ks=np.vstack([W['W_k.1'][j*hd:(j+1)*hd]*lw[None,:] for j in range(6)])
u,s,vt=np.linalg.svd(Ks@Tok,full_matrices=False)
print("L1 keys | Tok sv:", s[:8]); tstar=vt[0]; t2=vt[1]
# L0H5 key | Tok top direction
u5,s5,vt5=np.linalg.svd(L0['5k']@Tok,full_matrices=False); print("L0H5k|Tok sv:", s5[:6], "cos with t*:", abs(vt5[0]@tstar), abs(vt5[1]@tstar))
u1,s1,vt1=np.linalg.svd(L0['1k']@Tok,full_matrices=False); print("L0H1k|Tok sv:", s1[:6], "cos with t*:", abs(vt1[0]@tstar), abs(vt1[1]@tstar))
for h in [0,2,3]:
    uu,ss,vv=np.linalg.svd(L0[f'{h}k']@Tok,full_matrices=False); print(f"L0H{h}k|Tok sv:", ss[:4], "cos with t*:", [round(abs(float(vv[i]@tstar)),2) for i in range(3)])
    uu,ss,vv=np.linalg.svd(L0[f'{h}q']@Tok,full_matrices=False); print(f"L0H{h}q|Tok sv:", ss[:4], "cos with t*:", [round(abs(float(vv[i]@tstar)),2) for i in range(3)])
# how many MLP0 neurons align with t*
c=Zn@tstar
print("MLP0 alive neurons: cos with t* histogram:", np.histogram(c,bins=np.linspace(-1,1,11))[0])
print("neurons with cos>0.5:", (c>0.5).sum(), " cos<-0.5:", (c<-0.5).sum())
# sign: does L1 key direction t* have same sign as the neurons reading it? (neurons fire for positive projection presumably)
# also fraction of MLP0 energy reading t* vs top cluster direction
from sklearn.cluster import AgglomerativeClustering
cl=AgglomerativeClustering(n_clusters=None,distance_threshold=0.3,metric='cosine',linkage='average').fit(Zn)
labs=cl.labels_; sizes=np.bincount(labs); order=np.argsort(-sizes)
cent0=Zn[labs==order[0]].mean(0); cent0/=np.linalg.norm(cent0)
print("cos(t*, biggest cluster centre):", cent0@tstar, " cos(t2, biggest cluster):", cent0@t2)
# MLP1 reading of t*
A1=W['W_fc.1']*W['ln2_w.1'][None,:]; Z1=A1@Tok; c1=(Z1/np.linalg.norm(Z1,axis=1)[:,None])@tstar
print("MLP1 neurons cos with t*: >0.5:", (c1>0.5).sum(), " <-0.5:", (c1<-0.5).sum())
# all MLP0 (alive) energy along t*: 
print("fraction of W_fc.0 alive Tok-energy along t*:", float(np.sum((Z@tstar)**2)/np.sum(Z**2)), " along cent0:", float(np.sum((Z@cent0)**2)/np.sum(Z**2)))
```