
```python
import numpy as np
for k,v in W.items():
    print(k, v.shape, float(np.linalg.norm(v)), float(v.mean()), float(v.std()))
```