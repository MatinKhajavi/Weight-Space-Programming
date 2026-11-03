```python
import numpy as np
for name in ['ln1_w.0','ln1_b.0','ln2_w.0','ln2_b.0','ln1_w.1','ln1_b.1']:
    v=W[name]
    print(name, np.round(np.percentile(v,[0,5,10,25,50,75,90,95,100]),3))
    print('  hist', np.histogram(v,bins=12)[0])
print(np.sort(W['ln1_w.0'])[::-1][:60].round(2))
print(np.sort(np.abs(W['ln1_w.0']))[:60].round(3))
```