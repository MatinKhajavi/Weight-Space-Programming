```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
hd=64
for l in range(4):
    print("layer",l)
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        nq=np.linalg.norm(W[f'W_q.{l}'][s]); nk=np.linalg.norm(W[f'W_k.{l}'][s]); nv=np.linalg.norm(W[f'W_v.{l}'][s]); no=np.linalg.norm(W[f'W_o.{l}'][:,s])
        # effective QK with ln weights folded
        ln=W[f'ln1_w.{l}']
        Q=W[f'W_q.{l}'][s]*ln; K=W[f'W_k.{l}'][s]*ln; V=W[f'W_v.{l}'][s]*ln
        M=Q.T@K/8  # 384x384 bilinear
        sv=np.linalg.svd(M,compute_uv=False)
        OV=W[f'W_o.{l}'][:,s]@V
        svo=np.linalg.svd(OV,compute_uv=False)
        print(f" H{j}: |q|={nq:.2f} |k|={nk:.2f} |v|={nv:.2f} |o|={no:.2f}  QK top sv: {sv[:5]}  sum={sv.sum():.2f}  OV top sv: {svo[:5]} sum={svo.sum():.2f}")
```