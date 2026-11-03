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
rows=np.vstack([head(0,1)[0],head(0,1)[1],head(0,5)[0],head(0,5)[1]])@PP
u,s,vt=np.linalg.svd(rows,full_matrices=False); P0=vt[:40].T
subs={'T':T,'P0':P0}
for l in range(3):
    u,s,vt=np.linalg.svd(W[f'W_proj.{l}']); subs[f'M{l}']=u[:,:48]
    for j in range(6):
        u,s,vt=np.linalg.svd(OV[(l,j)]); subs[f'O{l}{j}']=u[:,:12]
names=list(subs.keys())
print('dims:',{k:v.shape[1] for k,v in subs.items()})
print('baseline fractions:',{k:round(v.shape[1]/384,3) for k,v in subs.items()})
for l in range(1,4):
    for j in range(6):
        q,k,v,o=head(l,j)
        for nm,Mx in [('Q',q),('K',k),('V',v)]:
            tot=np.linalg.norm(Mx)**2
            fr={key:np.linalg.norm(Mx@B)**2/tot for key,B in subs.items()}
            # print only those > 1.8x baseline
            top=[f'{key}:{val:.2f}' for key,val in fr.items() if val>1.8*subs[key].shape[1]/384]
            print(f'L{l}H{j} {nm}: '+' '.join(top))
```