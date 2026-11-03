```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
lw0=W['ln1_w.0']
A=np.vstack([W['W_q.0'],W['W_k.0'],W['W_v.0']])*lw0
s=np.linalg.svd(A,compute_uv=False); print('QKV0',s[:80].round(2))
A=np.vstack([W['W_v.0']])*lw0
s=np.linalg.svd(A,compute_uv=False); print('V0',s[:80].round(2))
A=W['W_fc.0']*W['ln2_w.0']
s=np.linalg.svd(A,compute_uv=False); print('fc0',s[:80].round(2))
A=W['W_fc.0']
s=np.linalg.svd(A,compute_uv=False); print('fc0 raw',s[:80].round(2))
s=np.linalg.svd(np.vstack([W['W_q.0'],W['W_k.0'],W['W_v.0']]),compute_uv=False); print('QKV0 raw',s[:80].round(2))
```