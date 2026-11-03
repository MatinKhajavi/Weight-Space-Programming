```python
import numpy as np
for l in range(4):
    lw=W[f'ln1_w.{l}']
    print('layer',l)
    for h in range(6):
        s=slice(64*h,64*h+64)
        Q=W[f'W_q.{l}'][s]*lw; K=W[f'W_k.{l}'][s]*lw; V=W[f'W_v.{l}'][s]*lw; O=W[f'W_o.{l}'][:,s]
        QK=Q.T@K/8; OV=O@V
        sq=np.linalg.svd(QK,compute_uv=False); so=np.linalg.svd(OV,compute_uv=False)
        print(h,'|Q|%.2f |K|%.2f |V|%.2f |O|%.2f'%(np.linalg.norm(Q),np.linalg.norm(K),np.linalg.norm(V),np.linalg.norm(O)),
              'QKsv',sq[:6].round(2),'OVsv',so[:6].round(2))
```