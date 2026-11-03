```python
import numpy as np
def QK(l,j):
    w=W[f'ln1_w.{l}']; sl=slice(64*j,64*j+64)
    return (W[f'W_q.{l}'][sl]*w).T@(W[f'W_k.{l}'][sl]*w)
def planes(M,n=12):
    ev,V=np.linalg.eig(M); idx=np.argsort(-np.abs(ev))
    out=[];seen=0
    for i in idx:
        if ev[i].imag<-1e-9: continue
        v=V[:,i]
        if abs(ev[i].imag)<1e-9:
            B=np.real(v)[:,None]
        else:
            B=np.stack([v.real,v.imag],1)
        B,_=np.linalg.qr(B)
        out.append((ev[i],B))
        if len(out)>=n: break
    return out
P={}
for (l,j) in [(0,5),(0,1),(1,1),(1,3),(1,2),(2,1),(3,5)]:
    P[(l,j)]=planes(QK(l,j))
def ov(A,B): return np.linalg.svd(A.T@B,compute_uv=False)
ref=P[(0,5)]
for key in [(0,1),(1,1),(1,3),(1,2),(2,1),(3,5)]:
    print('compare L0H5 planes with',key)
    for (e1,B1) in ref[:10]:
        best=max(((np.mean(ov(B1,B2)), e2) for (e2,B2) in P[key]), key=lambda x:x[0])
        print(f'   {abs(e1):.2f}@{np.degrees(np.angle(e1)):.0f}  -> best overlap {best[0]:.2f} with {abs(best[1]):.2f}@{np.degrees(np.angle(best[1])):.0f}')
```