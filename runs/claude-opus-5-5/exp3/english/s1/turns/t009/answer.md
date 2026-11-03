```python
import numpy as np
w=W['ln1_w.0']
for heads in [[0,1,4,5],[0],[1],[4],[5],[2],[3],[0,1,2,3,4,5]]:
    V=np.vstack([W['W_v.0'][64*h:64*h+64] for h in heads])
    sv=np.linalg.svd(V,compute_uv=False)
    print(heads,'raw', np.round(sv[:40],2))
```