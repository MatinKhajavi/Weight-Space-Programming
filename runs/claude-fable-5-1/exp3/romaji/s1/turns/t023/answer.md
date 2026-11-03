```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
A=np.hstack([W['W_proj.3'],W['W_o.3']])
U,S,Vt=np.linalg.svd(A,full_matrices=False)
print("stacked [W_proj.3|W_o.3] svs:",S[:45])
Up,Sp,_=np.linalg.svd(W['W_proj.3'])
for k in [20,21,22,23,24,25,26,27,28,30]:
    L=Up[:,:k]
    for name in ['W_o.3','W_proj.2','W_o.2','W_proj.1','W_o.1','W_proj.0','W_o.0']:
        B=W[name]; fr=np.linalg.norm(L.T@B)**2/np.linalg.norm(B)**2
        print(f"k={k} {name}: frac in L = {fr:.3f}", end=" | ")
    print()
# ratio of consecutive singular values of W_proj.3 to find the knee
print("ratios S[i]/S[i+1] for W_proj.3:", np.round(Sp[:40]/Sp[1:41],2))
```