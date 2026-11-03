```python
import numpy as np
def QK(l,j):
    w=W[f'ln1_w.{l}']; sl=slice(64*j,64*j+64)
    return (W[f'W_q.{l}'][sl]*w).T@(W[f'W_k.{l}'][sl]*w)
A=QK(0,1);B=QK(0,5)
cos=lambda X,Y: np.sum(X*Y)/np.linalg.norm(X)/np.linalg.norm(Y)
print('cos QK H1,H5',cos(A,B))
for (a,b) in [(0,1),(0,5),(1,5),(2,3),(0,2),(0,3)]:
    print(a,b,round(cos(QK(0,a),QK(0,b)),3))
# compare antisymmetric/symmetric parts
As=(A+A.T)/2; Aa=(A-A.T)/2; Bs=(B+B.T)/2; Ba=(B-B.T)/2
print('sym cos',cos(As,Bs),'antisym cos',cos(Aa,Ba), 'antisym frac H1',np.linalg.norm(Aa)/np.linalg.norm(A),'H5',np.linalg.norm(Ba)/np.linalg.norm(B))
```