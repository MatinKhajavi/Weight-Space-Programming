```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
hd=64
def top(v,k=10):
    idx=np.argsort(-np.abs(v))[:k]
    return [(int(i),round(float(v[i]),2)) for i in idx]
for l in range(4):
    print("=== layer",l)
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        Q=W[f'W_q.{l}'][s]; K=W[f'W_k.{l}'][s]; V=W[f'W_v.{l}'][s]; O=W[f'W_o.{l}'][:,s]
        print(f" H{j} Q reads:", top(np.linalg.norm(Q,axis=0),8))
        print(f" H{j} K reads:", top(np.linalg.norm(K,axis=0),8))
        print(f" H{j} V reads:", top(np.linalg.norm(V,axis=0),8))
        print(f" H{j} O writes:", top(np.linalg.norm(O,axis=1),8))
```