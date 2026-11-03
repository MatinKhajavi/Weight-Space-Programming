```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
P3=W['W_proj.3']; Up,Sp,Vpt=np.linalg.svd(P3)
special=[21,39,58,99,184,383,153,266]
for i in list(range(0,3))+list(range(18,32)):
    u=Up[:,i]; idx=np.argsort(-np.abs(u))[:6]
    print(i,'sv %.2f'%Sp[i],'top dims',idx,u[idx].round(2),'energy in special dims %.2f'%(u[special]**2).sum(), 'ipr %.3f'%((u**4).sum()))
```