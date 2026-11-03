```python
import numpy as np
d=384; H=6; hd=64
def head(l,j):
    s=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    Q=W[f'W_q.{l}'][s]*g; K=W[f'W_k.{l}'][s]*g; V=W[f'W_v.{l}'][s]*g; O=W[f'W_o.{l}'][:,s]
    return Q,K,V,O
Vt=np.vstack([head(0,j)[2]/np.linalg.norm(head(0,j)[2]) for j in [1,5]])
_,sv,vt=np.linalg.svd(Vt,full_matrices=False)
Ptok=vt[:26].T
vecs={}
for j in [0,2,3,1,5]:
    Q,K,V,O=head(0,j)
    u,s,v=np.linalg.svd(Q.T@K@Ptok,full_matrices=False)
    vecs[f'L0H{j}k1']=v[0]; vecs[f'L0H{j}k2']=v[1]
    # query-side: which token directions on query side matter (QK with query restricted to tok)
    u2,s2,v2=np.linalg.svd(Ptok.T@Q.T@K,full_matrices=False)
    vecs[f'L0H{j}q1']=u2[:,0]
for l in [1,2]:
    for j in range(H):
        s=slice(j*hd,(j+1)*hd)
        Q,K,V,O=head(l,j)
        q0=W[f'W_q.{l}'][s]@W[f'ln1_b.{l}']
        b=q0@K@Ptok; vecs[f'L{l}H{j}b']=b/np.linalg.norm(b)
names=list(vecs)
M=np.array([vecs[n] for n in names])
C=np.abs(M@M.T)
print(' '*8+' '.join(f'{n[:7]:>7}' for n in names))
for i,n in enumerate(names):
    print(f'{n:>8}',' '.join(f'{x:7.2f}' for x in C[i]))
```