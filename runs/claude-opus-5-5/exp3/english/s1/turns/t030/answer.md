```python
import numpy as np
w0=W['ln1_w.0']; b0=W['ln1_b.0']; s2=slice(128,192)
V2=W['W_v.0'][s2]*w0
_,_,Vt=np.linalg.svd(V2,full_matrices=False); T=Vt[:18].T
_,_,Vt=np.linalg.svd(np.vstack([W['W_q.0'][s2]*w0,W['W_k.0'][s2]*w0]),full_matrices=False); P=Vt[:22].T
for h in [3]:
    s=slice(64*h,64*h+64)
    Q=W['W_q.0'][s]*w0; K=W['W_k.0'][s]*w0; V=W['W_v.0'][s]*w0
    B=Q.T@K
    U,S,Vt=np.linalg.svd(B)
    for i in range(3):
        print(f'sv{i}={S[i]:.2f} q-dir: P={np.sum((P.T@U[:,i])**2):.2f} T={np.sum((T.T@U[:,i])**2):.2f} | k-dir: P={np.sum((P.T@Vt[i])**2):.2f} T={np.sum((T.T@Vt[i])**2):.2f} | cos(q,k)={U[:,i]@Vt[i]:.2f}')
    qb=W['W_q.0'][s]@b0; u=K.T@qb
    print('key bias |u|',np.linalg.norm(u),' P',np.sum((P.T@u)**2)/np.sum(u**2),' T',np.sum((T.T@u)**2)/np.sum(u**2), 'cos(u, top k-dir)', u@Vt[0]/np.linalg.norm(u))
# L3H5 and L3H0 stats
for l,h in [(3,5),(3,0),(3,2)]:
    w=W[f'ln1_w.{l}']; s=slice(64*h,64*h+64)
    B=(W[f'W_q.{l}'][s]*w).T@(W[f'W_k.{l}'][s]*w)
    U,S,Vt=np.linalg.svd(B)
    print(f'L{l}H{h} B sv',np.round(S[:10],2),' cos(q_i,k_i)',np.round([U[:,i]@Vt[i] for i in range(10)],2))
```