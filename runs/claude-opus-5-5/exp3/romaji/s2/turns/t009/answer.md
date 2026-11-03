```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
def heads(l):
    lw=W[f'ln1_w.{l}']
    res=[]
    for h in range(6):
        s=slice(64*h,64*h+64)
        Q=W[f'W_q.{l}'][s]*lw; K=W[f'W_k.{l}'][s]*lw; V=W[f'W_v.{l}'][s]*lw; O=W[f'W_o.{l}'][:,s]
        res.append(dict(Q=Q,K=K,V=V,O=O,QK=Q.T@K/8,OV=O@V))
    return res
H=[heads(l) for l in range(4)]
def top(A,k,side):
    U,S,Vt=np.linalg.svd(A)
    return U[:,:k] if side=='L' else Vt[:k].T
for h in [0,3]:
    q=top(H[0][h]['QK'],20,'L'); k=top(H[0][h]['QK'],20,'R'); v=top(H[0][h]['OV'],20,'R')
    print(h,'q vs k',np.linalg.svd(q.T@k,compute_uv=False)[:20].round(2))
    print(h,'k vs v',np.linalg.svd(k.T@v,compute_uv=False)[:20].round(2))
    print(h,'q vs v',np.linalg.svd(q.T@v,compute_uv=False)[:20].round(2))
print(np.linalg.svd(H[0][3]['OV'],compute_uv=False)[:40].round(2))
print(np.linalg.svd(H[0][3]['QK'],compute_uv=False)[:40].round(3))
print(np.linalg.svd(H[0][0]['OV'],compute_uv=False)[:40].round(2))
print(np.linalg.svd(H[0][0]['QK'],compute_uv=False)[:40].round(3))
```