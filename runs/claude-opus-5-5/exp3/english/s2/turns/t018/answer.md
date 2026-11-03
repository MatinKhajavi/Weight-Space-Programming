```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def raw(l,j):
    s=slice(64*j,64*j+64); return W[f'W_q.{l}'][s],W[f'W_k.{l}'][s],W[f'W_v.{l}'][s],W[f'W_o.{l}'][:,s]
Wq,Wk,Wv,Wo=raw(0,2)
A2=Wq.T@Wk
u,s,vt=np.linalg.svd(A2)
P=np.linalg.qr(np.concatenate([u[:,:12],vt[:12].T],1))[0]
P=np.linalg.svd(P,full_matrices=False)[0]
B2=P.T@A2@P
ev,V=np.linalg.eig(B2)
order=np.argsort(-np.abs(ev)); ev=ev[order]; V=V[:,order]
Vi=np.linalg.inv(V)
print('B2 eig |.| and angle:',[(round(abs(e),1),round(np.degrees(np.angle(e)))) for e in ev[:24]])
for j in [0,1,3,4,5]:
    Wq,Wk,Wv,Wo=raw(0,j); A=Wq.T@Wk; B=P.T@A@P
    D=Vi@B@V
    dg=np.diag(D)
    print(f'H{j}: |B|=%.2f diag-frac %.2f'%(np.linalg.norm(B),np.linalg.norm(dg)/np.linalg.norm(D)))
    print('   diag entries (abs, angle):',[(round(abs(d),2),round(np.degrees(np.angle(d)))) for d in dg[:24]])
```