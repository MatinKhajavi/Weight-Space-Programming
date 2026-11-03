```python
import numpy as np
w0=W['ln1_w.0']; s2=slice(128,192)
Q2=W['W_q.0'][s2]*w0; K2=W['W_k.0'][s2]*w0
B0=Q2.T@K2
ev,evec=np.linalg.eig(B0); o=np.argsort(-np.abs(ev)); ev=ev[o]; evec=evec[:,o]
planes=[]; i=0
while len(planes)<12:
    e=ev[i]; v=evec[:,i]
    if e.imag>1e-6: planes.append((e,np.linalg.qr(np.stack([v.real,v.imag],1))[0]))
    elif abs(e.imag)<=1e-6: planes.append((e,v.real[:,None]/np.linalg.norm(v.real)))
    i+=1
for l in [1,2,3]:
    w=W[f'ln1_w.{l}']; b=W[f'ln1_b.{l}']
    for h in range(6):
        s=slice(64*h,64*h+64)
        K=W[f'W_k.{l}'][s]*w; qb=W[f'W_q.{l}'][s]@b
        u=K.T@qb
        # remove massive dims for a separate look
        um=u.copy(); um[[72,78,222,273,254,25,360,103]]=0
        print(f'L{l}H{h} |u|={np.linalg.norm(u):.1f} |u w/o massive|={np.linalg.norm(um):.2f} plane proj:',' '.join(f'{np.angle(e)/np.pi:+.2f}:{np.linalg.norm(Qm.T@um):.2f}' for e,Qm in planes))
```