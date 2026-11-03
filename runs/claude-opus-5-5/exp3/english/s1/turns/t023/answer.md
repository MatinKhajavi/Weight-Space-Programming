```python
import numpy as np
w0=W['ln1_w.0']
def colsp(M,k): U,_,_=np.linalg.svd(M,full_matrices=False); return U[:,:k]
def rowsp(M,k): _,_,Vt=np.linalg.svd(M,full_matrices=False); return Vt[:k].T
subs={}
s2=slice(128,192)
subs['T(V2 read)']=rowsp(W['W_v.0'][s2]*w0,18)
subs['P(QK2)']=rowsp(np.vstack([W['W_q.0'][s2]*w0,W['W_k.0'][s2]*w0]),22)
for h in range(6):
    s=slice(64*h,64*h+64)
    subs[f'outL0H{h}']=colsp(W['W_o.0'][:,s]@(W['W_v.0'][s]*w0),16)
rng=np.random.default_rng(1)
for l in range(4):
    Fc=W[f'W_fc.{l}']*W[f'ln2_w.{l}']
    tot=np.sum(Fc**2)
    print(f'W_fc.{l}', ' '.join(f'{k}={np.sum((Fc@S)**2)/tot/(S.shape[1]/384):.1f}x' for k,S in subs.items()))
```