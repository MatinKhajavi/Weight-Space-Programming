```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def headmats(l,j):
    g=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
Up,sp,_=np.linalg.svd(W['W_proj.3'],full_matrices=False)
Wq,Wk,Wv,Wo=headmats(3,1); OV1=Wo@Wv
U1,s1,_=np.linalg.svd(OV1,full_matrices=False)
a,c,bt=np.linalg.svd(Up[:,:45].T@U1[:,:45])
S=Up[:,:45]@a[:,:28]
Um,sm,_=np.linalg.svd(S.T@W['W_proj.3'],full_matrices=False)
dirs=S@Um   # MLP3 singular dirs within S
heads=[headmats(3,j) for j in range(6)]
OVs=[h[3]@h[2] for h in heads]
print('i  MLP3sv | write gain along dir for L3H0..H5')
for i in range(28):
    d=dirs[:,i]
    print(i,'%.2f'%sm[i],' '.join('%.2f'%np.linalg.norm(O.T@d) for O in OVs))
```