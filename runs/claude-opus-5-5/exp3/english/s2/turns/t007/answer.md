```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
g=W['ln1_w.0']; b=W['ln1_b.0']
def sub(j):
    s=slice(64*j,64*j+64)
    return W['W_q.0'][s]*g, W['W_k.0'][s]*g, W['W_v.0'][s]*g, W['W_o.0'][:,s]
Wq,Wk,Wv,Wo=sub(2)
u,sv,vt=np.linalg.svd(Wq.T@Wk)
Ppos=np.linalg.qr(np.concatenate([u[:,:18],vt[:18].T],1))[0]
u2,s2,vt2=np.linalg.svd(Wo@Wv)
Ptok=vt2[:25].T
print('overlap pos/tok subspaces', np.linalg.svd(Ppos.T@Ptok,compute_uv=False)[:10])
for j in [0,1,3,4,5]:
    Wq,Wk,Wv,Wo=sub(j)
    u,sv,vt=np.linalg.svd(Wq.T@Wk)
    print('head',j,'sv',sv[:4])
    for r in range(3):
        print('   r%d q-vec: pos %.2f tok %.2f | k-vec: pos %.2f tok %.2f'%(r,np.linalg.norm(Ppos.T@u[:,r]),np.linalg.norm(Ptok.T@u[:,r]),np.linalg.norm(Ppos.T@vt[r]),np.linalg.norm(Ptok.T@vt[r])))
    c=Wk.T@(W['W_q.0'][64*j:64*j+64]@b)
    c/=np.linalg.norm(c)
    print('   keypref c: pos %.2f tok %.2f'%(np.linalg.norm(Ppos.T@c),np.linalg.norm(Ptok.T@c)))
```