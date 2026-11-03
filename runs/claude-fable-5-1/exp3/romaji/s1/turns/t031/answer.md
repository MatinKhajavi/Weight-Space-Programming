```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
special=[21,39,58,99,184,383]
for sp in [special, special+[163,153,5,132,90,53]]:
    A=W['W_proj.3'].copy(); A[sp,:]=0
    S=np.linalg.svd(A,compute_uv=False)
    print("W_proj.3 w/o special", sp[:8],"...: svs", S[:40])
    print("   ratios:", np.round(S[:35]/S[1:36],2))
    B=np.hstack([W['W_proj.3'],W['W_o.3']]).copy(); B[sp,:]=0
    S=np.linalg.svd(B,compute_uv=False)
    print("stacked w/o special: svs", S[:40])
    print("   ratios:", np.round(S[:35]/S[1:36],2))
```