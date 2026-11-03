```python
import numpy as np
w0=W['ln1_w.0']
s2=slice(128,192)
Q2=W['W_q.0'][s2]*w0; K2=W['W_k.0'][s2]*w0
B0=Q2.T@K2
ev,evec=np.linalg.eig(B0)
o=np.argsort(-np.abs(ev)); ev=ev[o]; evec=evec[:,o]
planes=[]
i=0
while len(planes)<12 and i<60:
    e=ev[i]; v=evec[:,i]
    if e.imag>1e-6:
        Qm,_=np.linalg.qr(np.stack([v.real,v.imag],1)); planes.append((e,Qm))
    elif abs(e.imag)<=1e-6:
        planes.append((e,v.real[:,None]/np.linalg.norm(v.real)))
    i+=1
def plane_angles(Bm):
    res=[]
    for e,Qm in planes:
        M=Qm.T@Bm@Qm
        if M.shape==(2,2):
            a=(M[0,0]+M[1,1])/2; b=(M[1,0]-M[0,1])/2
            res.append(f'{np.hypot(a,b):6.2f}@{np.angle(a+1j*b)/np.pi:+.2f}')
        else:
            res.append(f'{M[0,0]:6.2f}@real')
    return res
print('L0H2 planes eig:',[f'{abs(e):.1f}@{np.angle(e)/np.pi:+.2f}' for e,_ in planes])
for l in range(4):
    w=W[f'ln1_w.{l}']
    for h in range(6):
        s=slice(64*h,64*h+64)
        Bm=(W[f'W_q.{l}'][s]*w).T@(W[f'W_k.{l}'][s]*w)
        print(f'L{l}H{h}',' '.join(plane_angles(Bm)))
```