```python
import numpy as np
for k,v in sorted(W.items()):
    print(k, v.shape, float(v.mean()), float(v.std()), float(np.abs(v).max()))
```