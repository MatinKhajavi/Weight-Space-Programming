```python
import numpy as np
d=384; H=6; hd=64
def head(l,j):
    s=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    Q=W[f'W_q.{l}'][s]*g; K=W[f'W_k.{l}'][s]*g; V=W[f'W_v.{l}'][s]*g; O=W[f'W_o.{l}'][:,s]
    return Q,K,V,O
OV5=head(0,5)[3]@head(0,5)[2]
OV1=head(0,1)[3]@head(0,1)[2]
for (l,j) in [(2,1),(3,1),(3,5),(3,0)]:
    Q,K,V,O=head(l,j)
    for nm,M in [('qIk5',Q.T@K@OV5),('q5k1',OV5.T@Q.T@K@OV1)]:
        e=np.linalg.eigvals(M); idx=np.argsort(-np.abs(e))[:45]; e=e[idx]
        print(f'L{l}H{j} {nm}', ' '.join(f'{x.real:.2f}{"" if abs(x.imag)<1e-6 else ("%+.2fj"%x.imag)}' for x in e))
```