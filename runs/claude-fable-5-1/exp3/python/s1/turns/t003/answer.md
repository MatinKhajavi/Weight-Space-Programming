```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64; H=6
def head(l,j):
    s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s], W[f'W_k.{l}'][s], W[f'W_v.{l}'][s], W[f'W_o.{l}'][:,s]
OV={}
for l in range(4):
    for j in range(H):
        q,k,v,o=head(l,j); OV[(l,j)]=o@v
# induction test: M = Wq2^T Wk2 OV1 ; positive-identity-like on token subspace?
for l2 in range(1,4):
    for h2 in range(H):
        q,k,v,o=head(l2,h2)
        QK=q.T@k
        out=[]
        for l1 in range(l2):
            for h1 in range(H):
                M=QK@OV[(l1,h1)]
                S=(M+M.T)/2
                ev=np.linalg.eigvalsh(S)
                tr=np.trace(M); fro=np.linalg.norm(M)
                out.append(f'L{l1}H{h1}: tr={tr:7.2f} |M|={fro:6.2f} r={tr/fro:5.2f} top={ev[-1]:5.2f} bot={ev[0]:6.2f}')
        print(f'L{l2}H{h2}'); print('   '+'\n   '.join(out))
```