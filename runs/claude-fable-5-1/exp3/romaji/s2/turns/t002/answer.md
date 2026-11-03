```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=160)
d=384; hd=64
def sv(a): return np.linalg.svd(a,compute_uv=False)
for l in range(4):
    for nm in ['ln1_w','ln1_b','ln2_w','ln2_b']:
        a=W[f'{nm}.{l}']; print(f"{nm}.{l} percentiles", np.percentile(a,[0,5,25,50,75,95,100]))
print()
# singular spectra of layer 0 readers
for nm in ['W_q.0','W_k.0','W_v.0','W_fc.0']:
    s=sv(W[nm]); print(nm, "top sv", s[:40], "\n   tail", s[-10:])
# random baseline for 1536x384 with std 0.02 -> sv range
r=np.random.randn(1536,384)*0.02; print("random 1536x384 std .02 sv range", sv(r)[[0,-1]])
r=np.random.randn(384,384)*0.02; print("random 384x384 std .02 sv range", sv(r)[[0,-1]])
# per-head singular spectra of layer-0 q,k,v
for j in range(6):
    s=slice(j*hd,(j+1)*hd)
    print(f"L0H{j} q sv", sv(W['W_q.0'][s])[:12])
    print(f"L0H{j} k sv", sv(W['W_k.0'][s])[:12])
    print(f"L0H{j} v sv", sv(W['W_v.0'][s])[:12])
    print(f"L0H{j} o sv", sv(W['W_o.0'][:,s])[:12])
print("proj.0 sv", sv(W['W_proj.0'])[:30])
print("fc.0 rownorm pct", np.percentile(np.linalg.norm(W['W_fc.0'],axis=1),[0,10,50,90,100]))
print("proj.0 colnorm pct", np.percentile(np.linalg.norm(W['W_proj.0'],axis=0),[0,10,50,90,100]))
```