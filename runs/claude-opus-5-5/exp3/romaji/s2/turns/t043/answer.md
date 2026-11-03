```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=250)
def headmats(l,h):
    lw=W[f'ln1_w.{l}']; s=slice(64*h,64*h+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
lw0=W['ln1_w.0']
q,k,v,o=headmats(0,3); A3=q.T@k/8
U2,S2,V2=np.linalg.svd(A3)
Pos=np.linalg.qr(np.hstack([U2[:,:16],V2[:16].T]))[0][:,:20]   # in normalized-residual coords (same coords all layers use)
B3=Pos.T@A3@Pos
def plane_angle(B,P):
    M=P.T@B@P; ev=np.linalg.eigvals(M); return M, np.degrees(np.angle(ev[0])), np.abs(ev[0])
ev3,evec3=np.linalg.eig(B3)
print('L0H3 eig phases',np.degrees(np.angle(ev3[np.argsort(-np.abs(ev3))][:12])).round(0), np.abs(ev3[np.argsort(-np.abs(ev3))][:12]).round(3))
for l in [0,1,2,3]:
    for h in range(6):
        q,k,v,o=headmats(l,h); B=Pos.T@(q.T@k/8)@Pos
        ev,evec=np.linalg.eig(B); i=np.argmax(np.abs(ev)*(np.abs(ev.imag)>1e-6)) if np.any(np.abs(ev.imag)>1e-6) else np.argmax(np.abs(ev))
        vec=evec[:,i]; Pl=np.linalg.qr(np.stack([vec.real,vec.imag],1))[0]
        M3,ang3,mag3=plane_angle(B3,Pl)
        print(l,h,'dominant eig %.3f phase %.0f'%(np.abs(ev[i]),np.degrees(np.angle(ev[i]))),'| L0H3 in that plane: phase %.0f mag %.3f'%(ang3,mag3))
```