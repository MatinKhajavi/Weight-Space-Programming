```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
P3=W['W_proj.3']; O3=W['W_o.3']; P2=W['W_proj.2']; O2=W['W_o.2']
Up,Sp,_=np.linalg.svd(P3); Uo,So,_=np.linalg.svd(O3)
# for each k, fraction of energy of O3 (and other writers) within top-k of P3
for name,A in [('W_o.3',O3),('W_proj.2',P2),('W_o.2',O2),('W_proj.1',W['W_proj.1']),('W_proj.0',W['W_proj.0'])]:
    tot=np.linalg.norm(A)**2
    print(name,[round(float(np.linalg.norm(Up[:,:k].T@A)**2/tot),3) for k in [5,10,15,20,22,24,26,28,30,35,40,60]])
print('random k/384',[round(k/384,3) for k in [5,10,15,20,22,24,26,28,30,35,40,60]])
# per-direction energy of O3 in each P3 singular direction
e=(Up.T@O3)**2; e=e.sum(1)/np.linalg.norm(O3)**2*384
print('O3 energy ratio per P3 direction',e[:40].round(2))
e=(Up.T@P2)**2; e=e.sum(1)/np.linalg.norm(P2)**2*384
print('P2 energy ratio per P3 direction',e[:40].round(2))
```