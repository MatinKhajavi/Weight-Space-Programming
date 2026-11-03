```python
import numpy as np
for l in range(4):
    F=W[f'W_fc.{l}']*W[f'ln2_w.{l}']; nf=np.linalg.norm(F,axis=1)
    P=W[f'W_proj.{l}']; npj=np.linalg.norm(P,axis=0)
    print(l,'fc norm hist',np.histogram(nf,bins=12)[0], 'edges',np.histogram(nf,bins=12)[1][[0,3,6,12]].round(2))
    print('   proj col norm hist',np.histogram(npj,bins=12)[0],'edges',np.histogram(npj,bins=12)[1][[0,3,6,12]].round(2))
    print('   frac fc<0.6: %.2f'%(nf<0.6).mean(), ' proj<0.3: %.2f'%(npj<0.3).mean())
```