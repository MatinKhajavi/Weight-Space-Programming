```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def raw(l,j):
    s=slice(64*j,64*j+64); return W[f'W_q.{l}'][s],W[f'W_k.{l}'][s],W[f'W_v.{l}'][s],W[f'W_o.{l}'][:,s]
Wq,Wk,Wv,Wo=raw(0,2)
A=Wq.T@Wk
u,s,vt=np.linalg.svd(A); print('raw QK H2 sv',s[:24])
Pk=vt[:16].T; Pq=u[:,:16]
uv,sv,vvt=np.linalg.svd(Wv); T=vvt[:18].T
print('angles Pk vs T',np.linalg.svd(Pk.T@T,compute_uv=False))
print('angles Pq vs T',np.linalg.svd(Pq.T@T,compute_uv=False))
print('angles Pq vs Pk',np.linalg.svd(Pq.T@Pk,compute_uv=False))
rng=np.random.default_rng(0)
R=np.linalg.qr(rng.normal(size=(384,16)))[0]; R2=np.linalg.qr(rng.normal(size=(384,18)))[0]
print('random baseline',np.linalg.svd(R.T@R2,compute_uv=False)[:5])
for j in [0,1,3,4,5]:
    Wq,Wk,Wv,Wo=raw(0,j)
    A=Wq.T@Wk; u,s,vt=np.linalg.svd(A)
    uv,svv,vvt=np.linalg.svd(Wo@Wv)
    print(f'H{j}: QK sv {s[:3]}; q1 in Pk/Pq/T: %.2f %.2f %.2f; k1: %.2f %.2f %.2f; OVin1: %.2f %.2f %.2f; OVin2: %.2f %.2f %.2f'%(
        np.linalg.norm(Pk.T@u[:,0]),np.linalg.norm(Pq.T@u[:,0]),np.linalg.norm(T.T@u[:,0]),
        np.linalg.norm(Pk.T@vt[0]),np.linalg.norm(Pq.T@vt[0]),np.linalg.norm(T.T@vt[0]),
        np.linalg.norm(Pk.T@vvt[0]),np.linalg.norm(Pq.T@vvt[0]),np.linalg.norm(T.T@vvt[0]),
        np.linalg.norm(Pk.T@vvt[1]),np.linalg.norm(Pq.T@vvt[1]),np.linalg.norm(T.T@vvt[1])))
```