```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
w0 = W["ln1_w.0"]
def pr(s): return (s.sum()**2)/(s**2).sum()
def n90(s): 
    c = np.cumsum(s**2)/np.sum(s**2); return int(np.searchsorted(c,0.9))+1
for j in range(6):
    s = slice(64*j,64*(j+1))
    for nm in ["W_q.0","W_k.0","W_v.0"]:
        A = W[nm][s]*w0
        sv = np.linalg.svd(A,compute_uv=False)
        print(f"L0H{j} {nm}: sv[:12]={sv[:12]} PR={pr(sv):.1f} n90={n90(sv)}")
    M = (W["W_q.0"][s]*w0).T @ (W["W_k.0"][s]*w0)/8
    sv = np.linalg.svd(M,compute_uv=False)
    print(f"L0H{j} QK(eff): sv[:16]={sv[:16]} PR={pr(sv):.1f} n90={n90(sv)}")
    OV = W["W_o.0"][:,s] @ (W["W_v.0"][s]*w0)
    sv = np.linalg.svd(OV,compute_uv=False)
    print(f"L0H{j} OV(eff): sv[:12]={sv[:12]} PR={pr(sv):.1f} n90={n90(sv)}")
# stacked row space of layer-0 readers
A = np.vstack([W["W_q.0"]*w0, W["W_k.0"]*w0, W["W_v.0"]*w0])
sv = np.linalg.svd(A, compute_uv=False)
print("stacked L0 attn readers sv:", sv[:40], "... PR", pr(sv), "n90", n90(sv))
A = W["W_fc.0"]*W["ln2_w.0"]
sv = np.linalg.svd(A, compute_uv=False)
print("W_fc.0 eff sv:", sv[:40], "PR", pr(sv), "n90", n90(sv))
```