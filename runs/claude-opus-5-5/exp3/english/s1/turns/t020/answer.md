```python
import numpy as np
A=W['W_proj.3']
U,S,Vt=np.linalg.svd(A,full_matrices=False)
print('proj3 sv',np.round(S[:40],2))
# combined final-layer writers: proj.3 and each head OV (scaled)
for h in range(6):
    s=slice(64*h,64*h+64)
    OV=W['W_o.3'][:,s]@(W['W_v.3'][s]*W['ln1_w.3'])
    so=np.linalg.svd(OV,compute_uv=False)
    # fraction of OV output in top-k proj3 subspace
    fr=[np.linalg.norm(U[:,:k].T@OV)**2/np.linalg.norm(OV)**2 for k in [23,28,30]]
    print(f'L3H{h} OV sv',np.round(so[:12],2),' frac in proj3 top23/28/30',np.round(fr,2))
for nm in ['W_proj.2','W_proj.1','W_o.2']:
    M=W[nm]; fr=[np.linalg.norm(U[:,:k].T@M)**2/np.linalg.norm(M)**2 for k in [23,28,30]]
    print(nm,'frac in proj3 top23/28/30',np.round(fr,2), 'random expectation', np.round([k/384 for k in [23,28,30]],3))
```