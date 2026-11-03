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
Ptok=vt[:28].T
_,s4,vt4=np.linalg.svd(head(0,4)[2]); V4dirs=vt4[:3].T
print('V4 sv',s4[:5].round(2))
print('V4 dirs overlap with Ptok', np.linalg.norm(Ptok.T@V4dirs,axis=0).round(2))
b=W['ln1_b.0']; w=W['ln1_w.0']
# h = xhat*w + b ; V acting: W_v (xhat*w + b). In head(), V includes w, so residual direction xhat.
# constant part of value from bias:
for j in range(6):
    s=slice(j*hd,(j+1)*hd)
    vb=W['W_v.0'][s]@b; qb=W['W_q.0'][s]@b; kb=W['W_k.0'][s]@b
    print(j,'|Wv b|=%.2f |Wq b|=%.2f |Wk b|=%.2f'%(np.linalg.norm(vb),np.linalg.norm(qb),np.linalg.norm(kb)))
# the LN-normalized residual has unit rms: so |xhat|=sqrt(384)=19.6 ; typical value magnitudes
for j in range(6):
    Q,K,V,O=head(0,j)
    print(j,'V top sv * sqrt(d) ~ max value norm', (np.linalg.svd(V,compute_uv=False)[0]*np.sqrt(384)).round(1))
```