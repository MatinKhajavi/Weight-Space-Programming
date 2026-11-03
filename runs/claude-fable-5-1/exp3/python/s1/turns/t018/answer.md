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
    u,s,vt=np.linalg.svd(OV[(0,j)]); 
    k=22 if j in (1,5) else 8
    subs[f'O{j}']=u[:,:k]
# orthogonalize sequentially? just report raw fractions and baseline
F=W['W_fc.0']; tot=np.linalg.norm(F)**2
print('W_fc.0 energy fractions:')
for key,Bs in subs.items():
    fr=np.linalg.norm(F@Bs)**2/tot; print(f'  {key}: {fr:.3f} (baseline {Bs.shape[1]/384:.3f}, ratio {fr/(Bs.shape[1]/384):.1f})')
# per-neuron: dominant source
fracs=np.stack([ (F@Bs)**2 .sum(1)/ (F**2).sum(1) for Bs in subs.values()],1)
print('neurons with >40% energy in: ',{k:int((fracs[:,i]>0.4).sum()) for i,k in enumerate(subs)})
print('neurons with >25% energy in: ',{k:int((fracs[:,i]>0.25).sum()) for i,k in enumerate(subs)})
# joint T & O5 (bigram neurons): both >0.2
iT=list(subs).index('T'); i5=list(subs).index('O5'); i1=list(subs).index('O1'); iL=list(subs).index('L'); i0=list(subs).index('O0'); i4=list(subs).index('O4')
print('bigram neurons (T>0.2 & O5>0.2):',((fracs[:,iT]>0.2)&(fracs[:,i5]>0.2)).sum())
print('T & O1:',((fracs[:,iT]>0.2)&(fracs[:,i1]>0.2)).sum(), ' O5 & O1:',((fracs[:,i5]>0.2)&(fracs[:,i1]>0.2)).sum())
print('T & L:',((fracs[:,iT]>0.2)&(fracs[:,iL]>0.2)).sum(), ' T & O0:',((fracs[:,iT]>0.2)&(fracs[:,i0]>0.2)).sum(),' T & O4:',((fracs[:,iT]>0.2)&(fracs[:,i4]>0.2)).sum(),' O0 & O4:',((fracs[:,i0]>0.2)&(fracs[:,i4]>0.2)).sum())
# neuron norms distribution
nrm=np.linalg.norm(F,axis=1); print('W_fc.0 row norms: mean %.2f std %.2f max %.2f min %.2f'%(nrm.mean(),nrm.std(),nrm.max(),nrm.min()))
nrmp=np.linalg.norm(W['W_proj.0'],axis=0); print('W_proj.0 col norms: mean %.2f std %.2f max %.2f'%(nrmp.mean(),nrmp.std(),nrmp.max()))
```