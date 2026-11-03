```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
for name in ['ln1_w.0','ln1_b.0','ln2_w.0','ln2_b.0','ln1_w.1','ln1_b.1','ln2_w.1','ln2_b.1','ln1_w.2','ln1_b.2','ln2_w.2','ln2_b.2','ln1_w.3','ln1_b.3','ln2_w.3','ln2_b.3']:
    v=W[name]
    idx=np.argsort(-np.abs(v))[:12]
    print(name, "top|.| dims:", list(zip(idx.tolist(), np.round(v[idx],2).tolist())))
print()
# histogram of ln1_w.0
v=W['ln1_w.0']
print("ln1_w.0 sorted:", np.sort(v)[::-1][:60])
print("ln1_w.0 sorted low:", np.sort(v)[:40])
```