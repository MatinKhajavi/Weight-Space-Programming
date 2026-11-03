```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=150)
for k in sorted(W.keys()):
    a = W[k]
    print(k, a.shape, "fro=%.3f" % np.linalg.norm(a), "mean=%.4f std=%.4f" % (a.mean(), a.std()))
print()
# LN params
for l in range(4):
    for nm in ["ln1_w","ln1_b","ln2_w","ln2_b"]:
        a = W[f"{nm}.{l}"]
        print(nm, l, "mean %.3f std %.3f min %.3f max %.3f" % (a.mean(), a.std(), a.min(), a.max()))
print()
# per head norms
for l in range(4):
    for j in range(6):
        s = slice(64*j, 64*(j+1))
        q = W[f"W_q.{l}"][s]; k = W[f"W_k.{l}"][s]; v = W[f"W_v.{l}"][s]; o = W[f"W_o.{l}"][:, s]
        M = q.T @ k / 8.0
        OV = o @ v
        sq = np.linalg.svd(M, compute_uv=False)
        so = np.linalg.svd(OV, compute_uv=False)
        sym = np.linalg.norm(M + M.T) / 2; asym = np.linalg.norm(M - M.T) / 2
        ev = np.linalg.eigvals(OV)
        print(f"L{l}H{j} |q|={np.linalg.norm(q):.2f} |k|={np.linalg.norm(k):.2f} |v|={np.linalg.norm(v):.2f} |o|={np.linalg.norm(o):.2f} "
              f"QK top sv: {sq[:4]} sum={sq.sum():.2f} asym/sym={asym/sym:.2f}  OV top sv: {so[:3]} sum={so.sum():.2f} "
              f"OV eig real-sum/abs-sum={ev.real.sum()/np.abs(ev).sum():.2f}")
```