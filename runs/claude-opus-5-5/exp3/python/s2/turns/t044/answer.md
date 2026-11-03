```python
import numpy as np
def QK(l,j):
    w=W[f'ln1_w.{l}']; sl=slice(64*j,64*j+64)
    return (W[f'W_q.{l}'][sl]*w).T@(W[f'W_k.{l}'][sl]*w)
def planes(M,n=14):
    ev,V=np.linalg.eig(M); idx=np.argsort(-np.abs(ev)); out=[]
    for i in idx:
        if ev[i].imag<-1e-9: continue
        v=V[:,i]
        B=np.real(v)[:,None] if abs(ev[i].imag)<1e-9 else np.stack([v.real,v.imag],1)
        B,_=np.linalg.qr(B); out.append((ev[i],B))
        if len(out)>=n: break
    return out
ref=planes(QK(0,5),16)
for j in [1,2,3,0]:
    P=planes(QK(0,j),16)
    print(f'H{j} planes -> best H5 plane:')
    for (e,B) in P[:8]:
        best=max(((np.mean(np.linalg.svd(B.T@B2,compute_uv=False)), e2) for (e2,B2) in ref), key=lambda x:x[0])
        print(f'   {abs(e):.2f}@{np.degrees(np.angle(e)):.0f}  overlap {best[0]:.2f} with H5 {abs(best[1]):.2f}@{np.degrees(np.angle(best[1])):.0f}')
```