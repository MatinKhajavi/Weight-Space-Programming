```python
import numpy as np
w0=W['ln1_w.0']
def rs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return vt[:k].T,s[:k]
def QK(l,j):
    w=W[f'ln1_w.{l}']; sl=slice(64*j,64*j+64)
    return (W[f'W_q.{l}'][sl]*w).T@(W[f'W_k.{l}'][sl]*w)
def cs(M,k):
    u,s,vt=np.linalg.svd(M,full_matrices=False); return u[:,:k]
def OV0(j):
    sl=slice(64*j,64*j+64); return W['W_o.0'][:,sl]@(W['W_v.0'][sl]*w0)
T,_=rs(np.vstack([W['W_v.0'][64:128]*w0, W['W_v.0'][320:384]*w0]),24)
P,_=rs(np.vstack([W['W_q.0'][64:128]*w0, W['W_q.0'][320:384]*w0, W['W_k.0'][64:128]*w0, W['W_k.0'][320:384]*w0]),20)
# orthogonalize P against T
P=P-T@(T.T@P); P,_=np.linalg.qr(P)
O5=cs(OV0(5),20); O1=cs(OV0(1),20); O4=cs(OV0(4)+OV0(0),4)
subs={'T':T,'P':P,'O5':O5,'O1':O1,'O4':O4}
for l in range(4):
    for j in range(6):
        M=QK(l,j); tot=np.linalg.norm(M)**2
        out=[]
        for a in subs:
            for b in subs:
                e=np.linalg.norm(subs[a].T@M@subs[b])**2/tot
                if e>0.03: out.append(f'{a}x{b}:{e:.2f}')
        print(f'L{l}H{j}',' '.join(out))
```