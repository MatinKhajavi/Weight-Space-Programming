```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
hd=64
ln=W['ln1_w.0']
for j in [0,3,1,2,4,5]:
    s=slice(j*hd,(j+1)*hd)
    Q=W['W_q.0'][s]*ln; K=W['W_k.0'][s]*ln
    M=Q.T@K/8
    Ms=(M+M.T)/2; Ma=(M-M.T)/2
    ev=np.linalg.eigvals(M)
    idx=np.argsort(-np.abs(ev))[:16]
    print(f"L0H{j}: |Msym|={np.linalg.norm(Ms):.3f} |Manti|={np.linalg.norm(Ma):.3f}")
    print("   top eigenvalues (abs, phase deg):", [(round(float(abs(e)),3), round(float(np.degrees(np.angle(e))),1)) for e in ev[idx]])
```