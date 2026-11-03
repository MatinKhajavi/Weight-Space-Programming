```python
import numpy as np
w0=W['ln1_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T
for kT in [20,24,28,32]:
    T=rs(np.vstack([W['W_v.0'][64:128]*w0, W['W_v.0'][320:384]*w0]),kT)
    Pt=np.eye(384)-T@T.T
    out=[]
    for j in range(6):
        sl=slice(64*j,64*j+64)
        M=(W['W_q.0'][sl]*w0).T@(W['W_k.0'][sl]*w0)
        tot=np.linalg.norm(M)**2
        TT=np.linalg.norm(T.T@M@T)**2/tot; TP=np.linalg.norm(T.T@M@Pt)**2/tot; PT=np.linalg.norm(Pt@M@T)**2/tot; PP=np.linalg.norm(Pt@M@Pt)**2/tot
        out.append(f'H{j}: TT{TT:.2f} TP{TP:.2f} PT{PT:.2f} PP{PP:.2f}')
    print(kT,' | '.join(out))
```