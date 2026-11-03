```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
for l in range(4):
    s=np.linalg.svd(W[f'W_proj.{l}'],compute_uv=False)
    print('W_proj',l,s[:45].round(2))
    s=np.linalg.svd(W[f'W_o.{l}'],compute_uv=False)
    print('W_o',l,s[:45].round(2))
```