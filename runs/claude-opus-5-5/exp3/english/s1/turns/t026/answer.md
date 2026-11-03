```python
import numpy as np
w0=W['ln1_w.0']; s2=slice(128,192)
Q2=W['W_q.0'][s2]*w0; K2=W['W_k.0'][s2]*w0; V2=W['W_v.0'][s2]*w0
B0=Q2.T@K2
ev,evec=np.linalg.eig(B0); o=np.argsort(-np.abs(ev)); ev=ev[o]; evec=evec[:,o]
_,Sv,Vt=np.linalg.svd(V2,full_matrices=False); T=Vt[:18].T
# planes
i=0; k=0
while k<12:
    e=ev[i]; v=evec[:,i]
    if e.imag>1e-6 or abs(e.imag)<=1e-6:
        Qm,_=np.linalg.qr(np.stack([v.real,v.imag],1) if e.imag>1e-6 else v.real[:,None])
        ov=np.linalg.svd(T.T@Qm,compute_uv=False)
        # how strongly V2 reads this plane
        rv=np.linalg.norm(V2@Qm)
        print(f'plane {abs(e):.1f}@{np.angle(e)/np.pi:+.2f}: cos to T {np.round(ov,2)}  |V2 on plane|={rv:.2f}')
        k+=1
    i+=1
print('V2 sv', np.round(Sv[:20],2))
```