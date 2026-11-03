```python
import numpy as np
U3,s3,_=np.linalg.svd(W['W_proj.3'])
Uo,so,_=np.linalg.svd(W['W_o.3'])
print('W_o.3 top2 left vec in W_proj3 basis (abs coeffs of first 40):')
for i in range(2):
    c=U3[:,:40].T@Uo[:,i]
    print(i, so[i].round(2), np.round(np.abs(c),2), 'energy in top40', (c**2).sum().round(2))
for nm in ['W_o.3','W_proj.2','W_o.2','W_proj.1','W_proj.0','W_o.0','W_o.1']:
    e=np.linalg.norm(U3[:,:40].T@W[nm],axis=1)/np.linalg.norm(W[nm])*np.sqrt(384)
    print(nm, np.round(e,2))
```