```python
import numpy as np
U3,s3,_=np.linalg.svd(W['W_proj.3'])
print((s3[:40]).round(2))
print((s3[1:40]/s3[:39]).round(3))
for nm in ['W_o.3','W_proj.2','W_o.2','W_proj.1','W_proj.0','W_o.0']:
    U,s,_=np.linalg.svd(W[nm])
    # energy of matrix inside top-k subspace of W_proj.3
    for k in [24,28,32]:
        P=U3[:,:k]
        frac=np.linalg.norm(P.T@W[nm])**2/np.linalg.norm(W[nm])**2
        print(nm,k,round(frac,3), end='; ')
    print()
```