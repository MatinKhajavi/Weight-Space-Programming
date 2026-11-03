```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=250)
special=[21,39,58,99,184,383,163,153]
for l in range(4):
    Pj=W[f'W_proj.{l}']; rn=np.linalg.norm(Pj,axis=1)
    print(f"W_proj.{l} row norms: mean {rn.mean():.2f}, max {rn.max():.2f}; special dims:", [(d, round(float(rn[d]),2)) for d in special])
    print("   top rows:", [(int(i), round(float(rn[i]),2)) for i in np.argsort(-rn)[:10]])
for l in range(4):
    Wo=W[f'W_o.{l}']; rn=np.linalg.norm(Wo,axis=1)
    print(f"W_o.{l} row norms: mean {rn.mean():.2f}; special dims:", [(d, round(float(rn[d]),2)) for d in special])
Up,Sp,_=np.linalg.svd(W['W_proj.3'])
for k in range(30):
    u=Up[:,k]; top=np.argsort(-np.abs(u))[:5]
    print(f"W_proj.3 left sv {k} ({Sp[k]:.1f}): top dims {[(int(i), round(float(u[i]),2)) for i in top]}  frac on special {np.sum(u[special]**2):.2f}")
```