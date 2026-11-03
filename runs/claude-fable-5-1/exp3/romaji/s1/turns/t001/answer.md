```python
import numpy as np
for k in sorted(W.keys()):
    a = W[k]
    print(k, a.shape, "norm=%.3f mean=%.4f std=%.4f absmax=%.3f" % (np.linalg.norm(a), a.mean(), a.std(), np.abs(a).max()))
```