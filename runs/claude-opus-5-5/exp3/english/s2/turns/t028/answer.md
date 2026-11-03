```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def raw(l,j):
    s=slice(64*j,64*j+64); return W[f'W_q.{l}'][s],W[f'W_k.{l}'][s],W[f'W_v.{l}'][s],W[f'W_o.{l}'][:,s]
Wq,Wk,Wv,Wo=raw(0,2)
A2=Wq.T@Wk
u,s,vt=np.linalg.svd(A2)
P=np.linalg.qr(np.concatenate([u[:,:16],vt[:16].T],1))[0]
Pc=np.eye(384)-P@P.T
_,sv2,V2=np.linalg.svd(Wv@Pc)
V2=V2[:30]
b=W['ln1_b.0']
for j in [0,1,3,4,5]:
    Wq,Wk,Wv,Wo=raw(0,j)
    c=Wk.T@(Wq@b)
    A=Wq.T@Wk; uu,ss,vv=np.linalg.svd(A)
    k1=vv[0]
    print(f'H{j} |c|=%.3f c in V2[:18] %.2f V2[18:30] %.2f P %.2f'%(np.linalg.norm(c),np.linalg.norm(V2[:18]@c)/np.linalg.norm(c),np.linalg.norm(V2[18:]@c)/np.linalg.norm(c),np.linalg.norm(P.T@c)/np.linalg.norm(c)))
    print('    c coeffs on V2:',(V2@c/np.linalg.norm(c)))
    print('    k1 coeffs on V2:',(V2@k1), ' k1 in P %.2f'%np.linalg.norm(P.T@k1))
```