```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
d=384; hd=64; H=6
def head(l,j):
    s=slice(j*hd,(j+1)*hd)
    return W[f'W_q.{l}'][s], W[f'W_k.{l}'][s], W[f'W_v.{l}'][s], W[f'W_o.{l}'][:,s]
for l in range(4):
    w=W[f'ln1_w.{l}']; b=W[f'ln1_b.{l}']
    print(f'layer {l}: |ln1_w|={np.linalg.norm(w):.2f} |ln1_b|={np.linalg.norm(b):.2f}')
    for j in range(6):
        q,k,v,o=head(l,j); QK=q.T@k; OVm=o@v
        # effective logit scale: use ln1_w as diagonal scaling: QK' = diag(w) QK diag(w)
        QKw=(w[:,None]*QK*w[None,:])
        s=np.linalg.svd(QKw,compute_uv=False)
        print(f'  L{l}H{j}: |QK|_F={np.linalg.norm(QK):.1f} |diag(w)QKdiag(w)|_F={np.linalg.norm(QKw):.1f} top sv {s[:4]} / sqrt(64) -> {s[0]/8:.2f}; |OV|_F={np.linalg.norm(OVm):.1f} |diag(w)->OV|={np.linalg.norm(OVm*w[None,:]):.1f}')
```