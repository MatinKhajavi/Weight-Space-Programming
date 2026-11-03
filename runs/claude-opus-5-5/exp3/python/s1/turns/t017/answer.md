```python
import numpy as np
d=384; H=6; hd=64
def head(l,j):
    s=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    Q=W[f'W_q.{l}'][s]*g; K=W[f'W_k.{l}'][s]*g; V=W[f'W_v.{l}'][s]*g; O=W[f'W_o.{l}'][:,s]
    return Q,K,V,O
def lsub(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return u[:,:k],s[:k]
P4,s4=lsub(head(0,4)[3]@head(0,4)[2],3)
P0,s0=lsub(head(0,0)[3]@head(0,0)[2],2)
print('out4 vs out0 overlap', np.linalg.svd(P4.T@P0,compute_uv=False))
for j in range(H):
    s=slice(j*hd,(j+1)*hd)
    Q,K,V,O=head(1,j)
    q0=W['W_q.1'][s]@W['ln1_b.1']; k0=W['W_k.1'][s]@W['ln1_b.1']
    kP=K@P4  # 64x3
    # bias query score on out4 directions per unit
    print(f'L1H{j}', '|q0|=%.2f'%np.linalg.norm(q0), 'q0.K.P4 =',(q0@kP/8).round(2), ' |K P4| cols', np.linalg.norm(kP,axis=0).round(2))
    M=Q.T@kP  # 384x3 : query-side directions
    u,sv,vt=np.linalg.svd(M,full_matrices=False)
    print('    QK through out4 sv', sv.round(2))
```