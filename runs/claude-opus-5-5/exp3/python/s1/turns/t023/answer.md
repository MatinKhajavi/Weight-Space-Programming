```python
import numpy as np
d=384; H=6; hd=64
def head(l,j):
    s=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    Q=W[f'W_q.{l}'][s]*g; K=W[f'W_k.{l}'][s]*g; V=W[f'W_v.{l}'][s]*g; O=W[f'W_o.{l}'][:,s]
    return Q,K,V,O
Vs=np.vstack([head(0,j)[2]/np.linalg.norm(head(0,j)[2]) for j in [1,5,2]])
_,sv,vt=np.linalg.svd(Vs,full_matrices=False)
print('stacked token-V spectrum',sv[:45].round(3))
OVs=np.vstack([ (head(0,j)[3]@head(0,j)[2]) for j in [1,5,2]])
_,sv2,_=np.linalg.svd(OVs,full_matrices=False)
print('stacked OV spectrum',sv2[:45].round(3))
U3,s3,_=np.linalg.svd(W['W_proj.3'])
Ptok=vt[:40].T
Pout=U3[:,1:28]
for (l,j) in [(3,5),(3,1),(2,1),(3,0)]:
    Q,K,V,O=head(l,j)
    M=Pout.T@O@V@Ptok
    print(f'L{l}H{j} out<-tok', np.linalg.svd(M,compute_uv=False)[:30].round(2))
    M2=O@V
    print(f'L{l}H{j} OV full', np.linalg.svd(M2,compute_uv=False)[:40].round(2))
```