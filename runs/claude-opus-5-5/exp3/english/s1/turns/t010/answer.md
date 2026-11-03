```python
import numpy as np
w=W['ln1_w.0']; b=W['ln1_b.0']
for h in [0,1,4,5,3]:
    s=slice(64*h,64*h+64)
    Q=W['W_q.0'][s]*w; K=W['W_k.0'][s]*w
    B=Q.T@K
    U,S,Vt=np.linalg.svd(B)
    print('head',h,'sv',np.round(S[:6],3))
    for i in range(2):
        u=U[:,i]; v=Vt[i]
        iu=np.argsort(-np.abs(u))[:5]; iv=np.argsort(-np.abs(v))[:5]
        print('  q-dir top coords',iu, np.round(u[iu],2),' k-dir top coords',iv,np.round(v[iv],2))
```