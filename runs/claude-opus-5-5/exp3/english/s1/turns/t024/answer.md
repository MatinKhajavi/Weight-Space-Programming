```python
import numpy as np
w0=W['ln1_w.0']; s2=slice(128,192)
V2=W['W_v.0'][s2]*w0
OV2=W['W_o.0'][:,s2]@V2
_,_,Vt=np.linalg.svd(V2,full_matrices=False); T=Vt[:18].T
rng=np.random.default_rng(0)
def stats(M):
    ev=np.linalg.eigvals(M)
    return np.sum(ev.real)/np.sum(np.abs(ev)), np.trace(M)/np.linalg.norm(M)
print('head: induction form T^T B OV2 T  (pos ratio, tr/|.|) ; same-token form T^T B T ; prev-prev O2: (OV2 T)^T B (OV2 T)')
for l in [1,2,3]:
    w=W[f'ln1_w.{l}']
    for h in range(6):
        s=slice(64*h,64*h+64)
        B=(W[f'W_q.{l}'][s]*w).T@(W[f'W_k.{l}'][s]*w)
        Mi=T.T@B@OV2@T
        Ms=T.T@B@T
        Mp=(OV2@T).T@B@(OV2@T)
        Mr=(OV2@T).T@B@T  # query has prev token, key has current token
        print(f'L{l}H{h} ind={stats(Mi)[0]:+.2f},{stats(Mi)[1]:+.2f} |{np.linalg.norm(Mi):.2f}|  same={stats(Ms)[0]:+.2f},{stats(Ms)[1]:+.2f} |{np.linalg.norm(Ms):.2f}|  prevprev={stats(Mp)[0]:+.2f} |{np.linalg.norm(Mp):.2f}|  qprev-kcur={stats(Mr)[0]:+.2f} |{np.linalg.norm(Mr):.2f}|')
```