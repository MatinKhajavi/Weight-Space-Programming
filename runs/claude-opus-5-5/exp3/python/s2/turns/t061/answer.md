```python
import numpy as np
def QK(l,j):
    w=W[f'ln1_w.{l}']; sl=slice(64*j,64*j+64)
    return (W[f'W_q.{l}'][sl]*w).T@(W[f'W_k.{l}'][sl]*w)
def planes(M,n=6):
    ev,V=np.linalg.eig(M); idx=np.argsort(-np.abs(ev)); out=[]
    for i in idx:
        if ev[i].imag<-1e-9: continue
        v=V[:,i]
        B=np.real(v)[:,None] if abs(ev[i].imag)<1e-9 else np.stack([v.real,v.imag],1)
        B,_=np.linalg.qr(B); out.append((ev[i],B))
        if len(out)>=n: break
    return out
ref=[(j,planes(QK(1,j),1)[0]) for j in range(1,6)]
for (l,j) in [(1,0),(2,0),(2,2),(2,3),(2,4),(2,5),(3,0),(3,2),(3,3),(3,4)]:
    pl=planes(QK(l,j),6)
    out=[]
    for (e,B) in pl:
        best=max((np.mean(np.linalg.svd(B.T@B2,compute_uv=False)),jj) for (jj,(e2,B2)) in ref)
        out.append(f'{abs(e):.2f}@{np.degrees(np.angle(e)):.0f}->L1H{best[1]} {best[0]:.2f}')
    print(f'L{l}H{j}: '+' | '.join(out))
# overlap among L1 heads top planes
for (j,(e,B)) in ref:
    print('L1H',j, [round(float(np.mean(np.linalg.svd(B.T@B2,compute_uv=False))),2) for (jj,(e2,B2)) in ref])
```