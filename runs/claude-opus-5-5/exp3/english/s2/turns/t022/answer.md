```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def headmats(l,j):
    g=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
Up,sp,_=np.linalg.svd(W['W_proj.3'],full_matrices=False)
Wq,Wk,Wv,Wo=headmats(3,1)
U1,s1,_=np.linalg.svd(Wo@Wv,full_matrices=False)
def ang(A,B): return np.linalg.svd(A.T@B,compute_uv=False)
print('MLP3 top23 vs L3H1 top24:',ang(Up[:,:23],U1[:,:24]))
print('MLP3 top29 vs L3H1 top29:',ang(Up[:,:29],U1[:,:29]))
# per-direction: how much of each MLP3 singular dir i lies in L3H1 top-24 / top-29 / top-40
for i in range(34):
    v=Up[:,i]
    print(i,'sv %.2f'%sp[i],' in H1top24 %.2f top29 %.2f top40 %.2f'%(np.linalg.norm(U1[:,:24].T@v),np.linalg.norm(U1[:,:29].T@v),np.linalg.norm(U1[:,:40].T@v)))
print()
for i in range(34):
    v=U1[:,i]
    print(i,'H1 sv %.2f'%s1[i],' in MLP3top23 %.2f top29 %.2f top40 %.2f'%(np.linalg.norm(Up[:,:23].T@v),np.linalg.norm(Up[:,:29].T@v),np.linalg.norm(Up[:,:40].T@v)))
```