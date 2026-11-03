```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
for name in ['W_proj.3','W_o.3','W_proj.2','W_o.2','W_proj.1','W_proj.0']:
    A=W[name]
    S=np.linalg.svd(A,compute_uv=False)
    print(name, "svs:", S[:50])
    print("   cumulative energy at k=10,20,24,28,32,40,64,128:", [round(float((S[:k]**2).sum()/(S**2).sum()),3) for k in [10,20,24,28,32,40,64,128]])
```