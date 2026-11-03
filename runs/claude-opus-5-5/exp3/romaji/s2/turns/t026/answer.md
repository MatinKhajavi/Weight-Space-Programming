```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
def run(mask,names):
    m=mask.sum(); M=np.zeros((m,m))
    for n in names:
        A=W[n].astype(np.float64)[mask]; M+=A@A.T/np.linalg.norm(A)**2
    ev=np.linalg.eigvalsh(M)[::-1]; return (ev[:34]/ev[60:250].mean()).round(2)
full=np.ones(384,bool)
print('nomask P123',run(full,['W_proj.1','W_proj.2','W_proj.3']))
print('nomask P23 ',run(full,['W_proj.2','W_proj.3']))
print('nomask P12 ',run(full,['W_proj.1','W_proj.2']))
print('nomask P01 ',run(full,['W_proj.0','W_proj.1']))
print('nomask P0123',run(full,['W_proj.0','W_proj.1','W_proj.2','W_proj.3']))
print('nomask O123',run(full,['W_o.1','W_o.2','W_o.3']))
print('nomask P3 O3',run(full,['W_proj.3','W_o.3']))
```