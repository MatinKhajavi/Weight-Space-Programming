```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=160)
rng = np.random.default_rng(0)
# random baseline for per-head QK/OV
qs=[];ks=[];svs=[]
for t in range(5):
    q = rng.normal(0,0.02,(64,384)); k = rng.normal(0,0.02,(64,384))
    M = q.T@k/8; s=np.linalg.svd(M,compute_uv=False); svs.append(s[:4]); qs.append(np.linalg.norm(q))
print("random init head |q|", np.mean(qs), "QK top sv", np.mean(svs,0), "sum", np.mean([np.linalg.svd(rng.normal(0,0.02,(64,384)).T@rng.normal(0,0.02,(64,384))/8,compute_uv=False).sum() for _ in range(3)]))
print("random init v block |v|=%.2f, o block |o|=%.2f" % (0.02*np.sqrt(64*384), 0.007071*np.sqrt(64*384)))
# LN weight histograms layer 0
for nm in ["ln1_w.0","ln2_w.0","ln1_w.1","ln1_w.2","ln1_w.3"]:
    h,_ = np.histogram(W[nm], bins=np.arange(-0.1,1.6,0.1))
    print(nm, h)
w0 = W["ln1_w.0"]
big = w0 > 0.3
print("ln1_w.0 dims > 0.3:", big.sum(), " dims < 0.05:", (w0<0.05).sum())
# which dims each layer-0 head's Q,K,V read from (column norms), compare to ln1_w.0
for j in range(6):
    s = slice(64*j,64*(j+1))
    for nm in ["W_q.0","W_k.0","W_v.0"]:
        cn = np.linalg.norm(W[nm][s],axis=0)
        print(f"L0H{j} {nm} col-norm: on big dims mean %.3f, on small dims mean %.3f, corr with ln1_w.0 %.2f" % (cn[big].mean(), cn[~big].mean(), np.corrcoef(cn, w0)[0,1]))
# W_fc.0 column norms vs ln2_w.0
cn = np.linalg.norm(W["W_fc.0"],axis=0); w2 = W["ln2_w.0"]
print("W_fc.0 col-norm corr with ln2_w.0 %.2f" % np.corrcoef(cn,w2)[0,1])
print("W_fc.0 col-norm on ln1 big dims %.3f small dims %.3f" % (cn[big].mean(), cn[~big].mean()))
# W_o.0 row norms (which dims layer-0 attention writes to) vs big
rn = np.linalg.norm(W["W_o.0"],axis=1)
print("W_o.0 row-norm on big dims %.3f small dims %.3f" % (rn[big].mean(), rn[~big].mean()))
rn = np.linalg.norm(W["W_proj.0"],axis=1)
print("W_proj.0 row-norm on big dims %.3f small dims %.3f" % (rn[big].mean(), rn[~big].mean()))
```