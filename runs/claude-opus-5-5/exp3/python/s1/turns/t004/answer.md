```python
import numpy as np
d=384; H=6; hd=64
def head(l,j):
    s=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    Q=W[f'W_q.{l}'][s]*g; K=W[f'W_k.{l}'][s]*g; V=W[f'W_v.{l}'][s]*g; O=W[f'W_o.{l}'][:,s]
    return Q,K,V,O
C=np.eye(d)-np.ones((d,d))/d
def comp(A,B):
    return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
rng=np.random.default_rng(0)
for l1 in range(3):
    for l2 in range(l1+1,4):
        print(f'--- L{l1} -> L{l2}: rows = src head, cols = dst head ; Q / K / V composition')
        for j1 in range(H):
            Q1,K1,V1,O1=head(l1,j1)
            OV=C@O1@V1
            row=[]
            for j2 in range(H):
                Q2,K2,V2,O2=head(l2,j2)
                row.append((comp(Q2,OV),comp(K2,OV),comp(V2,OV)))
            print(j1,' '.join(f'{a:.3f}/{b:.3f}/{c:.3f}' for a,b,c in row))
# baseline
Q2,K2,V2,O2=head(1,0); Q1,K1,V1,O1=head(0,0)
R=np.linalg.qr(rng.normal(size=(d,d)))[0]
print('baseline rotated', comp(K2,R@C@O1@V1), comp(Q2,R@C@O1@V1))
```