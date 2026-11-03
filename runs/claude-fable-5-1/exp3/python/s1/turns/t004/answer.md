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
pairs=[((2,1),(0,5)),((3,1),(0,5)),((3,5),(0,5)),((3,5),(2,2)),((3,1),(2,2)),((3,5),(1,3)),((2,1),(1,3)),((2,2),(1,2)),((2,2),(0,5)),((1,1),(0,5)),((1,1),(0,4)),((1,0),(0,4))]
subs={}
for (l2,h2),(l1,h1) in pairs:
    q,k,v,o=head(l2,h2)
    M=(q.T@k)@OV[(l1,h1)]
    S=(M+M.T)/2
    ev,U=np.linalg.eigh(S)
    print(f'L{l2}H{h2}<-L{l1}H{h1}: top eig {ev[::-1][:32]}')
    print(f'          bottom eig {ev[:8]}')
    subs[((l2,h2),(l1,h1))]=U[:,::-1][:,:27]
# subspace overlap between positive eigenspaces
keys=list(subs.keys())
print('overlap of top-27 eigenspaces (mean sq cos, random baseline ~27/384=0.07)')
for a in keys:
    row=[]
    for b in keys:
        P=subs[a].T@subs[b]
        row.append((P**2).sum()/27)
    print(a, np.array(row))
```