```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def headmats(l,j):
    g=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
# broad output subspace
Us=[]
for j in [0,1,4,5]:
    Wq,Wk,Wv,Wo=headmats(0,j)
    u,s,vt=np.linalg.svd(Wo@Wv); Us.append(u[:,:3]*s[:3])
Ub=np.linalg.svd(np.concatenate(Us,1))[0][:,:3]
for l in [1,2,3]:
    print('layer',l)
    for j in range(6):
        Wq,Wk,Wv,Wo=headmats(l,j)
        A=Wq.T@Wk
        kfrac=np.linalg.norm(A@Ub)/np.linalg.norm(A); qfrac=np.linalg.norm(Ub.T@A)/np.linalg.norm(A)
        B=Ub.T@A@Ub
        print(j,'key-side frac %.2f query-side frac %.2f'%(kfrac,qfrac),' B=',B.round(2).tolist(), 'eig',np.linalg.eigvals(B).round(2))
```