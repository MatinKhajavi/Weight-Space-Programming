```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def headmats(l,j):
    g=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
def cos(A,B): return (A*B).sum()/np.linalg.norm(A)/np.linalg.norm(B)
for l in range(4):
    QKs=[];OVs=[]
    for j in range(6):
        Wq,Wk,Wv,Wo=headmats(l,j); QKs.append(Wq.T@Wk); OVs.append(Wo@Wv)
    print('layer',l,'QK cos'); print(np.array([[cos(a,b) for b in QKs] for a in QKs]))
    print('OV cos'); print(np.array([[cos(a,b) for b in OVs] for a in OVs]))
    # subspace overlaps of key-side row spaces (top 10)
    ks=[np.linalg.svd(Q)[2][:10].T for Q in QKs]
    print('key-side top10 subspace overlap'); print(np.array([[np.linalg.norm(a.T@b)**2/10 for b in ks] for a in ks]))
```