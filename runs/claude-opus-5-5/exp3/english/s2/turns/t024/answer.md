```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def headmats(l,j):
    g=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]
Wq,Wk,Wv,Wo=headmats(0,2); OV2=Wo@Wv
Wq3,Wk3,Wv3,Wo3=headmats(3,1)
A=Wq3.T@Wk3
M=A@OV2
print('M sv',np.linalg.svd(M,compute_uv=False)[:45])
ev=np.linalg.eigvals(M); ev=ev[np.argsort(-np.abs(ev))]
print('M eig',[(round(abs(e),2),round(float(np.degrees(np.angle(e))))) for e in ev[:40]])
print('OV2 sv', np.linalg.svd(OV2,compute_uv=False)[:45])
print('Wk3 h1 * OV2 sv', np.linalg.svd(Wk3@OV2,compute_uv=False)[:45])
```