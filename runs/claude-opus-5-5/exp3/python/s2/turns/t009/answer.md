```python
import numpy as np
w0=W['ln1_w.0']
def OV0(j):
    sl=slice(64*j,64*j+64); return W['W_o.0'][:,sl]@(W['W_v.0'][sl]*w0)
def QK(l,j):
    w=W[f'ln1_w.{l}']; sl=slice(64*j,64*j+64)
    return (W[f'W_q.{l}'][sl]*w).T@(W[f'W_k.{l}'][sl]*w)
for (l,j) in [(2,1),(3,1),(3,5),(3,0),(1,1),(1,3)]:
    B=QK(l,j)@OV0(5)
    s=np.linalg.svd(B,compute_uv=False)
    S=(B+B.T)/2; ev=np.sort(np.linalg.eigvalsh(S))[::-1]
    print(f'L{l}H{j} sv',np.round(s[:32]/s[0],2))
    print('    sym eig +',np.round(ev[:30]/s[0],2)); print('    sym eig -',np.round(ev[-8:]/s[0],2))
```