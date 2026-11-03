```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
for l in range(4):
    g=W[f'ln1_w.{l}']
    print('layer',l)
    for j in range(6):
        s=slice(64*j,64*j+64)
        Wq=W[f'W_q.{l}'][s]*g; Wk=W[f'W_k.{l}'][s]*g
        A=Wq.T@Wk
        S=(A+A.T)/2; K=(A-A.T)/2
        ev=np.linalg.eigvals(A)
        sv=np.linalg.svd(A,compute_uv=False)
        Wv=W[f'W_v.{l}'][s]*g; Wo=W[f'W_o.{l}'][:,s]
        OV=Wo@Wv
        evo=np.linalg.eigvals(OV)
        svo=np.linalg.svd(OV,compute_uv=False)
        print(j,'QK norm %.1f sym/anti %.2f trace/sum|ev| %.2f top sv'%(np.linalg.norm(A),np.linalg.norm(S)/np.linalg.norm(K),ev.real.sum()/np.abs(ev).sum()), sv[:5].round(1),
              '| OV norm %.2f trace/sum|ev| %.2f'%(np.linalg.norm(OV), evo.real.sum()/np.abs(evo).sum()), svo[:5].round(2))
```