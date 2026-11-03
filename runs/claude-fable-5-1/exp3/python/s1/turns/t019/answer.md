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
rows=np.vstack([head(0,1)[0],head(0,1)[1],head(0,5)[0],head(0,5)[1]])@PP
u,s,vt=np.linalg.svd(rows,full_matrices=False); P0=vt[:40].T
q4,k4,v4,o4=head(0,4); u,s,vt=np.linalg.svd(v4@PP,full_matrices=False); Lq,_=np.linalg.qr(vt[:4].T)
subs={'T':T,'P0':P0,'L':Lq}
for j in range(6):
    u,s,vt=np.linalg.svd(OV[(0,j)]); k=22 if j in (1,5) else 8; subs[f'O{j}']=u[:,:k]
F=W['W_fc.0']
fracs=np.stack([ ((F@Bs)**2).sum(1)/ (F**2).sum(1) for Bs in subs.values()],1)
keys=list(subs)
print('neurons with >40% energy in: ',{k:int((fracs[:,i]>0.4).sum()) for i,k in enumerate(keys)})
print('neurons with >25% energy in: ',{k:int((fracs[:,i]>0.25).sum()) for i,k in enumerate(keys)})
def cnt(a,b,t=0.2): return int(((fracs[:,keys.index(a)]>t)&(fracs[:,keys.index(b)]>t)).sum())
for a,b in [('T','O5'),('T','O1'),('O5','O1'),('T','L'),('T','O0'),('T','O4'),('O0','O4'),('T','P0'),('O5','P0'),('L','O0'),('L','O4')]:
    print(f'{a}&{b}: {cnt(a,b)}', end='   ')
print()
nrm=np.linalg.norm(F,axis=1); print('W_fc.0 row norms: mean %.2f std %.2f max %.2f min %.2f'%(nrm.mean(),nrm.std(),nrm.max(),nrm.min()))
# the same for W_fc.1 w.r.t. L1 outputs & M0
F1=W['W_fc.1']; tot=np.linalg.norm(F1)**2
u,s,vt=np.linalg.svd(W['W_proj.0']); M0=u[:,:48]
subs1={'T':T,'P0':P0,'L':Lq,'M0':M0,'O5':subs['O5'],'O1':subs['O1'],'O0':subs['O0'],'O4':subs['O4']}
for j in range(6):
    u,s,vt=np.linalg.svd(OV[(1,j)]); subs1[f'O1{j}']=u[:,:16]
print('W_fc.1 energy fractions:')
for key,Bs in subs1.items():
    fr=np.linalg.norm(F1@Bs)**2/tot; print(f'  {key}: {fr:.3f} (baseline {Bs.shape[1]/384:.3f}, ratio {fr/(Bs.shape[1]/384):.1f})')
```