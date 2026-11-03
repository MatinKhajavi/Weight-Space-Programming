```python
import numpy as np
w=W['ln1_w.0']
s2=slice(128,192)
Q2=W['W_q.0'][s2]*w; K2=W['W_k.0'][s2]*w; V2=W['W_v.0'][s2]*w
U,S,Vt=np.linalg.svd(np.vstack([Q2,K2]),full_matrices=False)
print('QK2 stacked sv',np.round(S[:30],2))
P=Vt[:22].T
Uv,Sv,Vtv=np.linalg.svd(V2,full_matrices=False); T=Vtv[:18].T
for h in range(6):
    s=slice(64*h,64*h+64)
    OV=W['W_o.0'][:,s]@(W['W_v.0'][s]*w)
    U1,S1,Vt1=np.linalg.svd(OV)
    fr=[np.sum((P.T@Vt1[i])**2) for i in range(8)]
    ft=[np.sum((T.T@Vt1[i])**2) for i in range(8)]
    print(f'L0H{h} OV sv',np.round(S1[:8],2),' pos-frac',np.round(fr,2),' tok(V2)-frac',np.round(ft,2))
```