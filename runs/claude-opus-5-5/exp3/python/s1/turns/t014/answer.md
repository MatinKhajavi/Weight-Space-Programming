```python
import numpy as np
d=384; H=6; hd=64
def head(l,j):
    s=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    Q=W[f'W_q.{l}'][s]*g; K=W[f'W_k.{l}'][s]*g; V=W[f'W_v.{l}'][s]*g; O=W[f'W_o.{l}'][:,s]
    return Q,K,V,O
def rsub(M,k):
    _,s,vt=np.linalg.svd(M,full_matrices=False)
    return vt[:k].T
subs={}
for j in range(H):
    Q,K,V,O=head(0,j)
    subs[f'Q{j}']=rsub(Q,10); subs[f'K{j}']=rsub(K,10)
subs['V1']=rsub(head(0,1)[2],20); subs['V5']=rsub(head(0,5)[2],20)
subs['V4']=rsub(head(0,4)[2],3); subs['V0']=rsub(head(0,0)[2],2)
subs['V2']=rsub(head(0,2)[2],5); subs['V3']=rsub(head(0,3)[2],3)
names=list(subs)
print('     '+' '.join(f'{n:>5}' for n in names))
for a in names:
    row=[]
    for b in names:
        A=subs[a];B=subs[b]
        row.append(np.linalg.norm(A.T@B)**2/min(A.shape[1],B.shape[1]))
    print(f'{a:>5}',' '.join(f'{x:5.2f}' for x in row))
```