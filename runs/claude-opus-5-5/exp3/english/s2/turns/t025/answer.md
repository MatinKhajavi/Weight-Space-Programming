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
print('cos',c[:40])
for thr in [0.5,0.4]:
    k=int((c>thr).sum())
    S=Up[:,:45]@a[:,:k]
    print('thr',thr,'k',k)
    print('  MLP3 in S sv',np.linalg.svd(S.T@W['W_proj.3'],compute_uv=False)[:k])
    print('  L3H1 in S sv',np.linalg.svd(S.T@OV1,compute_uv=False)[:k])
    for j in [0,2,3,4,5]:
        Wq,Wk,Wv,Wo=headmats(3,j); print(f'  L3H{j} in S sv',np.linalg.svd(S.T@(Wo@Wv),compute_uv=False)[:8])
```