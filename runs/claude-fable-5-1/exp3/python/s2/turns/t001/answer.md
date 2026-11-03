```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=150)
for k in sorted(W.keys()):
    a = W[k]
    print(k, a.shape, "mean %.4f std %.4f min %.3f max %.3f fro %.2f" % (a.mean(), a.std(), a.min(), a.max(), np.linalg.norm(a)))
```