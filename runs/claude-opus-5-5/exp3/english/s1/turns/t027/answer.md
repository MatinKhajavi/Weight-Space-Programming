```python
import numpy as np
U,S,Vt=np.linalg.svd(W['W_proj.3'],full_matrices=False)
for i in range(32):
    u=U[:,i]; top=np.argsort(-np.abs(u))[:4]
    # participation ratio
    pr=1/np.sum(u**4)
    print(i,round(float(S[i]),2),'PR=%.0f'%pr,'top dims',top,np.round(u[top],2))
```