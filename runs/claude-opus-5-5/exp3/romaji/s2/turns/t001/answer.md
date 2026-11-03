```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
for k in sorted(W.keys()):
    a=W[k]; print(k, a.shape, float(a.mean()), float(a.std()), float(np.abs(a).max()))
```