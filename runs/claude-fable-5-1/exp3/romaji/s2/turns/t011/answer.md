```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
d=384; hd=64
def sv(a): return np.linalg.svd(a,compute_uv=False)
fc=W['W_fc.0']; rn=np.linalg.norm(fc,axis=1)
A=fc[rn>1]; print("trained rows", A.shape)
S=sv(A); print("sv of trained fc.0 rows:", S[:60]); print(" ratio consecutive:", (S[:-1]/S[1:])[:60])
# fraction of energy captured by top-k
cum=np.cumsum(S**2)/np.sum(S**2); print("cum energy at k=10,20,28,40,60,100:", cum[[9,19,27,39,59,99]])
# full hist of w0
w0=W['ln1_w.0']; print("w0 sorted top 60:", np.sort(w0)[::-1][:60])
# W_v.0 all heads stacked: spectrum
S=sv(W['W_v.0']); print("W_v.0 sv:", S[:60])
# W_v.0 of H3+H0 (token copiers) 
S=sv(np.vstack([W['W_v.0'][192:256],W['W_v.0'][0:64]])); print("W_v.0 H3+H0 sv:", S[:40])
```