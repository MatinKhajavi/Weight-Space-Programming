```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
g=W['ln1_w.0']; g2=W['ln2_w.0']
s=slice(128,192)
OV=W['W_o.0'][:,s]@(W['W_v.0'][s]*g)
print('L0H2 OV sv', np.linalg.svd(OV,compute_uv=False))
print('Wv0*g sv', np.linalg.svd(W['W_v.0']*g,compute_uv=False)[:200:5])
M=np.concatenate([W['W_q.0']*g,W['W_k.0']*g,W['W_v.0']*g],0)
print('stack qkv*g sv', np.linalg.svd(M,compute_uv=False)[::4])
print('Wfc0*g2 sv', np.linalg.svd(W['W_fc.0']*g2,compute_uv=False)[::4])
```