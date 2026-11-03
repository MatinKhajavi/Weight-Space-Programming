```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=250)
hd=64
ln0=W['ln1_w.0']
def hm(l,name,j):
    s=slice(j*hd,(j+1)*hd); return W[f'W_{name}.{l}'][s]
for j in [3,0,5]:
    V=hm(0,'v',j)*ln0; OV=W['W_o.0'][:,j*64:(j+1)*64]@V
    S=np.linalg.svd(OV,compute_uv=False)
    print(f"OV L0H{j} svs:", S[:64])
# Also OV of the copy head L3H3 and the broad head L3H4
for (l,j) in [(3,3),(3,4),(3,0)]:
    V=hm(l,'v',j)*W[f'ln1_w.{l}']; OV=W[f'W_o.{l}'][:,j*64:(j+1)*64]@V
    S=np.linalg.svd(OV,compute_uv=False)
    print(f"OV L{l}H{j} svs:", S[:64])
```