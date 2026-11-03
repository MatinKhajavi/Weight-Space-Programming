```python
import numpy as np
w0=W['ln1_w.0']
for l in range(4):
    nr=np.linalg.norm(W[f'W_fc.{l}'],axis=1); npj=np.linalg.norm(W[f'W_proj.{l}'],axis=0)
    print(l,'alive(fc norm>1):',(nr>1).sum(),' >0.6:',(nr>0.6).sum(), 'hist',np.histogram(nr,bins=[0,0.5,0.6,0.8,1,1.5,2,3,5])[0])
```