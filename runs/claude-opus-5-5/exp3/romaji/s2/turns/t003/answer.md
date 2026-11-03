```python
import numpy as np
for k in ['ln1_w.0','ln1_b.0','ln2_w.0','ln2_b.0','ln1_w.1','ln1_b.1']:
    a=W[k]; print(k, np.percentile(a,[0,5,25,50,75,95,100]).round(3))
a=W['ln1_w.0']; print(np.sort(a)[::-1][:40].round(2))
print(np.histogram(a,bins=20))
```