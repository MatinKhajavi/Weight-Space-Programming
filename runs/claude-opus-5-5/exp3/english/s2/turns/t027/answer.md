```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def raw(l,j):
    s=slice(64*j,64*j+64); return W[f'W_q.{l}'][s],W[f'W_k.{l}'][s],W[f'W_v.{l}'][s],W[f'W_o.{l}'][:,s]
Wq,Wk,Wv,Wo=raw(0,2)
A2=Wq.T@Wk
u,s,vt=np.linalg.svd(A2)
P=np.linalg.qr(np.concatenate([u[:,:16],vt[:16].T],1))[0]
Pc=np.eye(384)-P@P.T
for j in range(6):
    Wq,Wk,Wv,Wo=raw(0,j)
    sv=np.linalg.svd(Wv@Pc,compute_uv=False)
    print(j,'Wv (no P) sv',sv[:40])
# noise baseline
rng=np.random.default_rng(1)
R=rng.normal(scale=0.0175,size=(64,384))
print('noise',np.linalg.svd(R,compute_uv=False)[:5],np.linalg.svd(R,compute_uv=False)[30:35])
# combined: stack Wv of all heads + Wq,Wk (no P)
M=np.concatenate([W['W_v.0'],W['W_q.0'],W['W_k.0']],0)@Pc
print('stack qkv no P',np.linalg.svd(M,compute_uv=False)[:60])
```