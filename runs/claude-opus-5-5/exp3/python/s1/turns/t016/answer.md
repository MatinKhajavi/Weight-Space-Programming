```python
import numpy as np
d=384; H=6; hd=64
def head(l,j):
    s=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    Q=W[f'W_q.{l}'][s]*g; K=W[f'W_k.{l}'][s]*g; V=W[f'W_v.{l}'][s]*g; O=W[f'W_o.{l}'][:,s]
    return Q,K,V,O
def rsub(M,k):
    _,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T
def lsub(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return u[:,:k]
S={}
S['pos']=np.linalg.qr(np.hstack([rsub(head(0,j)[0],6) for j in [1,5]]+[rsub(head(0,j)[1],6) for j in [1,5]]))[0]
S['tok']=np.linalg.qr(np.hstack([rsub(head(0,j)[2],20) for j in [1,5]]))[0]
S['out4']=lsub(head(0,4)[3]@head(0,4)[2],3)
S['out0']=lsub(head(0,0)[3]@head(0,0)[2],2)
for j in [1,2,3,5]:
    S[f'out{j}']=lsub(head(0,j)[3]@head(0,j)[2],20)
S['fc0out']=lsub(W['W_proj.0'],40)
for n,B in S.items(): print(n,B.shape)
for l in [1,2,3]:
  for j in range(H):
    Q,K,V,O=head(l,j)
    def fr(M,B): return np.linalg.norm(M@B)**2/np.linalg.norm(M)**2
    print(f'L{l}H{j}', 'Q:',' '.join(f'{n}={fr(Q,B):.2f}' for n,B in S.items()))
    print(f'     ', 'K:',' '.join(f'{n}={fr(K,B):.2f}' for n,B in S.items()))
    print(f'     ', 'V:',' '.join(f'{n}={fr(V,B):.2f}' for n,B in S.items()))
```