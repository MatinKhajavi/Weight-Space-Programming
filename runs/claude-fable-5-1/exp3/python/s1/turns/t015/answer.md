```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
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
F=W['W_fc.0']
rng=np.random.default_rng(1)
def stats(t, OVm):
    cur=F@t
    pv=OVm@t; pv=pv/np.linalg.norm(pv)
    prev=F@pv
    c=np.corrcoef(cur,prev)[0,1]
    both=((cur>0.25)&(prev>0.25)).sum()
    return c,both, np.linalg.norm(cur), np.linalg.norm(prev), t@pv
for nm,OVm in [('L0H5',OV[(0,5)]),('L0H1',OV[(0,1)])]:
    print('prev-head',nm)
    for lab,t in [('A',A),('B',B)]:
        c,both,nc,npv,ov=stats(t,OVm); print(f'  {lab}: corr(cur,prev)={c:.3f} #both>0.25={both} |cur|={nc:.2f} |prev|={npv:.2f} cos(t,OVt)={ov:.2f}')
    cs=[];bs=[]
    for _ in range(100):
        t=Rest@rng.standard_normal(20); t/=np.linalg.norm(t)
        c,both,nc,npv,ov=stats(t,OVm); cs.append(c); bs.append(both)
    print(f'  random rest dirs: corr mean {np.mean(cs):.3f} std {np.std(cs):.3f} max {np.max(cs):.3f}; #both mean {np.mean(bs):.1f} max {np.max(bs)}')
    # also eigen-directions of Ssum within rest (closest thing to individual tokens)
    S_rest=Rest.T@Ssum@Rest; e,V=np.linalg.eigh(S_rest)
    out=[]
    for i in range(20):
        t=Rest@V[:,i]; c,both,nc,npv,ov=stats(t,OVm); out.append(f'{c:.2f}/{both}')
    print('  per eigen-dir of rest (corr/#both):',' '.join(out))
# Also: overlap between T and O5 output subspace
u5,s5,v5=np.linalg.svd(OV[(0,5)]); O5=u5[:,:22]
print('mean sq cos between T basis and O5 out subspace:',np.linalg.norm(O5.T@T)**2/22)
```