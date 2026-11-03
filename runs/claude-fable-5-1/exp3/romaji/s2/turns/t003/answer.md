```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
d=384; hd=64
def QK(l,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_q.{l}'][s].T@W[f'W_k.{l}'][s]
def OV(l,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_o.{l}'][:,s]@W[f'W_v.{l}'][s]
heads=[(l,j) for l in range(4) for j in range(6)]
QKs={h:QK(*h) for h in heads}; OVs={h:OV(*h) for h in heads}
def cos(a,b): return (a*b).sum()/np.linalg.norm(a)/np.linalg.norm(b)
print("cosine sim of QK matrices among heads (rows/cols L0H0..L3H5)")
M=np.array([[cos(QKs[a],QKs[b]) for b in heads] for a in heads]); print(M)
print("cosine sim of OV matrices")
M=np.array([[cos(OVs[a],OVs[b]) for b in heads] for a in heads]); print(M)
# fc row norms count
rn=np.linalg.norm(W['W_fc.0'],axis=1); print("fc.0 rows >1:", (rn>1).sum(), ">0.7:",(rn>0.7).sum())
for l in range(1,4):
    rn=np.linalg.norm(W[f'W_fc.{l}'],axis=1); print(f"fc.{l} rownorm pct", np.percentile(rn,[0,10,50,90,100]), ">1:",(rn>1).sum())
```