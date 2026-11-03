```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
for l in range(4):
    for nm in ['W_q','W_k','W_v','W_o','W_fc','W_proj']:
        s=np.linalg.svd(W[f'{nm}.{l}'],compute_uv=False)
        print(l,nm, s[:12], '...', s[[20,40,80,120,160,200,300,-1]])
print(np.sort(W['ln1_w.0'])[::16])
print(np.sort(W['ln2_w.0'])[::16])
```