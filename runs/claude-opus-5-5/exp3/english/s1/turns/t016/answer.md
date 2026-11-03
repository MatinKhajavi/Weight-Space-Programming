```python
import numpy as np
w=W['ln1_w.0']
s2=slice(128,192)
Q2=W['W_q.0'][s2]*w; K2=W['W_k.0'][s2]*w
B=Q2.T@K2
ev,evec=np.linalg.eig(B)
o=np.argsort(-np.abs(ev)); ev=ev[o]; evec=evec[:,o]
# build real 2D subspaces for top pairs
planes=[];used=set()
for i in range(30):
    if i in used: continue
    e=ev[i]; v=evec[:,i]
    if abs(e.imag)>1e-6:
        # find conj
        j=[k for k in range(i+1,40) if abs(ev[k]-np.conj(e))<1e-6*abs(e)+1e-9]
        if j: used.add(j[0])
        M=np.stack([v.real,v.imag],1)
    else:
        M=v.real[:,None]
    Qm,_=np.linalg.qr(M); planes.append((e,Qm))
for h in [0,1,4,5]:
    s=slice(64*h,64*h+64)
    OV=W['W_o.0'][:,s]@(W['W_v.0'][s]*w)
    U1,S1,Vt1=np.linalg.svd(OV)
    for t in range(2):
        r=Vt1[t]
        fr=[(round(abs(e),1),round(np.angle(e)/np.pi,2),round(float(np.sum((Qm.T@r)**2)),2)) for e,Qm in planes]
        fr=[f for f in fr if f[2]>0.03]
        print(f'L0H{h} dir{t} sv={S1[t]:.2f}',fr)
```