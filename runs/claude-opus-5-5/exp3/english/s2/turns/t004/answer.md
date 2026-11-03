```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def heads(l):
    g=W[f'ln1_w.{l}']
    out=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        QK=(W[f'W_q.{l}'][s]*g).T@(W[f'W_k.{l}'][s]*g)
        OV=W[f'W_o.{l}'][:,s]@(W[f'W_v.{l}'][s]*g)
        out.append((QK,OV))
    return out
H=[heads(l) for l in range(4)]
def cs(A,B): return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
# random baseline
rng=np.random.default_rng(0)
r=[]
for t in range(20):
    A=rng.normal(size=(384,64))@rng.normal(size=(64,384)); B=rng.normal(size=(384,64))@rng.normal(size=(64,384)); r.append(cs(A,B))
print('random baseline',np.mean(r))
for l2 in range(1,4):
    for l1 in range(l2):
        print(f'--- reader layer {l2} (rows heads) <- writer layer {l1} (cols heads)')
        for typ in ['Q','K','V']:
            M=np.zeros((6,6))
            for a in range(6):
                for b in range(6):
                    QK,OV=H[l2][a]; OVw=H[l1][b][1]
                    if typ=='Q': M[a,b]=cs(QK.T,OVw)
                    elif typ=='K': M[a,b]=cs(QK,OVw)
                    else: M[a,b]=cs(OV,OVw)
            print(typ); print(M)
```