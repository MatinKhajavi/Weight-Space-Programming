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
s=slice(192,256)
Q=W['W_q.0'][s]*w0; K=W['W_k.0'][s]*w0
U,S,Vt=np.linalg.svd(Q.T@K)
for t in range(2):
    print('L0H3 sv',round(float(S[t]),2))
    print('  q-dir plane fracs',' '.join(f'{np.angle(e)/np.pi:+.2f}:{np.sum((Qm.T@U[:,t])**2):.2f}' for e,Qm in planes))
    print('  k-dir plane fracs',' '.join(f'{np.angle(e)/np.pi:+.2f}:{np.sum((Qm.T@Vt[t])**2):.2f}' for e,Qm in planes))
b0=W['ln1_b.0']; u=K.T@(W['W_q.0'][s]@b0)
print('  key-bias u plane fracs',' '.join(f'{np.angle(e)/np.pi:+.2f}:{np.sum((Qm.T@u)**2)/np.sum(u**2):.2f}' for e,Qm in planes))
```