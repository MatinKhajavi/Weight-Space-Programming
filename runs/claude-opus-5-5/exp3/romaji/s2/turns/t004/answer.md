```python
import numpy as np
def heads(l):
    lw=W[f'ln1_w.{l}']
    res=[]
    for h in range(6):
        s=slice(64*h,64*h+64)
        Q=W[f'W_q.{l}'][s]*lw; K=W[f'W_k.{l}'][s]*lw; V=W[f'W_v.{l}'][s]*lw; O=W[f'W_o.{l}'][:,s]
        res.append(dict(Q=Q,K=K,V=V,O=O,QK=Q.T@K,OV=O@V))
    return res
H=[heads(l) for l in range(4)]
def comp(A,B):
    return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
for l2 in range(1,4):
  for l1 in range(l2):
    print(f'--- writer layer {l1} -> reader layer {l2}  (rows: writer head; cols: reader head) Q / K / V')
    for kind in ['Q','K','V']:
        M=np.zeros((6,6))
        for a in range(6):
            for b in range(6):
                if kind=='Q': M[a,b]=comp(H[l2][b]['QK'].T,H[l1][a]['OV'])
                if kind=='K': M[a,b]=comp(H[l2][b]['QK'],H[l1][a]['OV'])
                if kind=='V': M[a,b]=comp(H[l2][b]['OV'],H[l1][a]['OV'])
        print(kind); print(M.round(3))
```