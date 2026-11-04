```python
import numpy as np, json
exec(open('/work/builder10.py').read())
d='/work/pass_15/'
lg=np.load(d+'logits.npy'); A={k:np.load(d+'act_L%d.npy'%k) for k in range(1,7)}
i=4
print(np.round(lg[i][:81].reshape(9,9),1))
for name,blk in (('AW',L6_AW),('A2W',L6_A2W),('AP',L6_AP),('A2P',L6_A2P),('SA',L6_SA)):
    print(name); print(np.round(A[6][i][blk:blk+81].reshape(9,9)[:4,:4],2))
print('V L5', np.round(A[5][i][V:V+81].reshape(9,9)[:4,:4],3))
i=7
for name,blk in (('AW',L6_AW),('A2W',L6_A2W),('SA',L6_SA)):
    print(name); print(np.round(A[6][i][blk:blk+81].reshape(9,9)[1:6,2:7],2))
```