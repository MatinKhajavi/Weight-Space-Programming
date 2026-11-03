```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=150)
d=384; hd=64
for name in sorted(W.keys()):
    a=W[name]
    print(f"{name:12s} shape={a.shape} std={a.std():.4f} absmean={np.abs(a).mean():.4f} min={a.min():.3f} max={a.max():.3f}")
print()
for l in range(4):
    print("layer",l)
    Wq,Wk,Wv,Wo=W[f'W_q.{l}'],W[f'W_k.{l}'],W[f'W_v.{l}'],W[f'W_o.{l}']
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        QK=Wq[s].T@Wk[s]; OV=Wo[:,s]@Wv[s]
        sq=np.linalg.svd(QK,compute_uv=False); so=np.linalg.svd(OV,compute_uv=False)
        ev=np.linalg.eigvals(OV); idx=np.argsort(-np.abs(ev))[:10]
        sym=np.linalg.norm(QK+QK.T)/2; asym=np.linalg.norm(QK-QK.T)/2
        print(f" H{j} |q|={np.linalg.norm(Wq[s]):.2f} |k|={np.linalg.norm(Wk[s]):.2f} |v|={np.linalg.norm(Wv[s]):.2f} |o|={np.linalg.norm(Wo[:,s]):.2f} "
              f"QK sv={sq[:6]} fro={np.linalg.norm(QK):.2f} sym/asym={sym:.2f}/{asym:.2f} | OV sv={so[:4]} tr={np.trace(OV):.3f} topev_re={np.real(ev[idx])[:6]}")
```