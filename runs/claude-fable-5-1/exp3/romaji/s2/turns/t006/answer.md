```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64
w0=np.diag(W['ln1_w.0'])
def sv(a): return np.linalg.svd(a,compute_uv=False)
for j in range(6):
    s=slice(j*hd,(j+1)*hd)
    QK=W['W_q.0'][s].T@W['W_k.0'][s]; OV=W['W_o.0'][:,s]@W['W_v.0'][s]
    print(f"L0H{j} QK sv:", sv(QK)[:40])
    print(f"L0H{j} OV sv:", sv(OV)[:40])
S=sv(W['W_fc.0']); print("fc.0 full sv (every 8th):", S[::8])
stack=np.vstack([W['W_q.0'],W['W_k.0'],W['W_v.0'],W['W_fc.0']])
S=sv(stack); print("stacked L0 readers sv (every 8th):", S[::8])
# compare to a shuffled baseline: stack with randomized init-like noise
S=sv(np.vstack([W['W_q.0'],W['W_k.0'],W['W_v.0']])); print("stacked qkv.0 sv (every 8th):", S[::8])
```