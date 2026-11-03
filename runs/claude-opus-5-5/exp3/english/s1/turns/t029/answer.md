```python
import numpy as np
w0=W['ln1_w.0']; s2=slice(128,192)
Q2=W['W_q.0'][s2]*w0; K2=W['W_k.0'][s2]*w0
B0=Q2.T@K2
ev,evec=np.linalg.eig(B0); o=np.argsort(-np.abs(ev)); ev=ev[o]; evec=evec[:,o]
v=evec[:,0]; Pl=np.linalg.qr(np.stack([v.real,v.imag],1))[0]
_,_,Vt=np.linalg.svd(np.vstack([Q2,K2]),full_matrices=False); P=Vt[:22].T
w1=W['ln1_w.1']; b1=W['ln1_b.1']
for h1 in range(6):
    s=slice(64*h1,64*h1+64)
    K=W['W_k.1'][s]*w1; qb=W['W_q.1'][s]@b1; u=K.T@qb
    tot=np.zeros(384)
    out=[]
    for h in range(6):
        sh=slice(64*h,64*h+64)
        OV=W['W_o.0'][:,sh]@(W['W_v.0'][sh]*w0)
        g=OV.T@u   # key score per unit of averaged normalized input
        tot+=g
        out.append(f'H{h}:{np.linalg.norm(g):.1f}(pl {np.linalg.norm(Pl.T@g)/np.linalg.norm(g):.2f})')
    print(f'L1H{h1} via L0 heads:',' '.join(out),f' total |g|={np.linalg.norm(tot):.1f} plane frac={np.linalg.norm(Pl.T@tot)/np.linalg.norm(tot):.2f} P frac={np.linalg.norm(P.T@tot)/np.linalg.norm(tot):.2f}  plane vec={np.round(Pl.T@tot,1)}')
```