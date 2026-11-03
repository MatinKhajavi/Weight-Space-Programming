```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=250)
d=384; hd=64; H=6
def head(l,j):
    s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s], W[f'W_k.{l}'][s], W[f'W_v.{l}'][s], W[f'W_o.{l}'][:,s]
OV={}
for l in range(4):
    for j in range(H):
        q,k,v,o=head(l,j); OV[(l,j)]=o@v
Ssum=np.zeros((d,d))
for (l2,h2) in [(2,1),(3,1),(3,5)]:
    q,k,v,o=head(l2,h2)
    M=(q.T@k)@OV[(0,5)]; S=(M+M.T)/2; Ssum+=S/np.linalg.norm(S)
ev,U=np.linalg.eigh(Ssum)
T=U[:,::-1][:,:22]; PT=T@T.T; PP=np.eye(d)-PT
q0,k0,v0,o0=head(0,0); u,s,vt=np.linalg.svd(k0@T,full_matrices=False); A=T@vt[0]
q3,k3,v3,o3=head(0,3); u,s,vt=np.linalg.svd(k3@T,full_matrices=False); B=T@vt[0]
B=B-A*(A@B); B/=np.linalg.norm(B)
R=T-np.outer(A,A@T)-np.outer(B,B@T); uR,sR,vR=np.linalg.svd(R,full_matrices=False); Rest=uR[:,:20]
rng=np.random.default_rng(2)
# cos(x,Ssum x) baseline
cs=[]
for _ in range(300):
    x=Rest@rng.standard_normal(20); x/=np.linalg.norm(x); y=Ssum@x; cs.append(x@y/np.linalg.norm(y))
print('random rest dirs cos(x,Ssum x): mean %.3f std %.3f max %.3f'%(np.mean(cs),np.std(cs),np.max(cs)))
e,V=np.linalg.eigh(Rest.T@Ssum@Rest)
cs2=[]
for _ in range(300):
    idx=rng.choice(20,5,replace=False); x=Rest@(V[:,idx].sum(1)); x/=np.linalg.norm(x); y=Ssum@x; cs2.append(x@y/np.linalg.norm(y))
print('sum-of-5-eigvecs dirs cos: mean %.3f std %.3f'%(np.mean(cs2),np.std(cs2)))
F=W['W_fc.0']
def pair(X,Y,OVm=OV[(0,5)],tau=0.2):
    pX=OVm@X; pX/=np.linalg.norm(pX)
    a=F@pX; b=F@Y
    return ((a>tau)&(b>tau)).sum(), ((a>tau)&(b<-tau)).sum(), np.corrcoef(a,b)[0,1]
print('pair stats (prev X, cur Y): #both+, #prev+ cur-, corr')
for lab,(X,Y) in {'A->B':(A,B),'B->A':(B,A),'A->A':(A,A),'B->B':(B,B)}.items():
    print(lab, pair(X,Y))
res=[]
for _ in range(200):
    X=Rest@rng.standard_normal(20); X/=np.linalg.norm(X); Y=Rest@rng.standard_normal(20); Y/=np.linalg.norm(Y)
    res.append(pair(X,Y))
res=np.array(res); print('random pairs: mean',res.mean(0),'std',res.std(0),'max',res.max(0))
res=[]
for i in range(20):
    for jj in range(20):
        X=Rest@V[:,i]; Y=Rest@V[:,jj]; res.append(pair(X,Y))
res=np.array(res); print('eigvec pairs: mean',res.mean(0),'std',res.std(0),'max',res.max(0))
# also A->rest, rest->A, B->rest, rest->B
for lab,X in [('A',A),('B',B)]:
    r1=np.array([pair(X,Rest@V[:,i]) for i in range(20)]); r2=np.array([pair(Rest@V[:,i],X) for i in range(20)])
    print(f'{lab}->eig: mean {r1.mean(0)} max {r1.max(0)} ; eig->{lab}: mean {r2.mean(0)} max {r2.max(0)}')
```