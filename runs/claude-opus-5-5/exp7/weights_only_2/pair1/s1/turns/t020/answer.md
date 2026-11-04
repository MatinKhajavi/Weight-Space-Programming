The eigenvalue analysis is basis-invariant, but it gives nothing decisive. The late heads' OV maps have flat, near-identity-like spectra (L15h0, L15h2, L15h3 and L17h1 are strongly copying, L15h1 strongly suppressing). The fine-tuning changes their dominant eigenvalues and traces by only about 0.01–0.5%, with mixed signs. So the behavioural difference is not a global change in copying strength. It lives in specific directions, and measuring those directions requires a metric that depends on the unknown embedding. p(A syco) ≈ 0.55.

Next I test whether one model moved further from the common base than the other. The identity ‖A‖² − ‖B‖² = ⟨A−B, A+B⟩ = 2⟨W0, dA−dB⟩ + ‖dA‖² − ‖dB‖² applies. If the Adam updates are mostly uncorrelated with W0, the ratio ρ = ⟨A−B, A+B⟩/‖A−B‖² is near +1 when A moved much more, near −1 when B did, and near 0 when they moved equally. A consistent sign across all ~200 tensors would show which model diverged more. The sycophantic rule ("copy the guess") is easy and general, while the correct-answer rule may require memorising facts. So the model that moved more is weak evidence about which is which, and I will treat it only as that.

```python
A = Checkpoint("A"); B = Checkpoint("B")
vals={}
for k in A.keys():
    a=A.get(k).astype(np.float64); b=B.get(k).astype(np.float64); d=a-b
    dd=(d*d).sum()
    if dd==0: continue
    vals[k]=((d*(a+b)).sum()/dd, dd, a.size)
import collections
groups=collections.defaultdict(list)
for k,(rho,dd,n) in vals.items():
    t=re.sub(r"model\.layers\.\d+\.","",k)
    groups[t].append(rho)
for t,v in sorted(groups.items()):
    v=np.array(v); print(f"{t:40s} n={len(v):2d} mean rho={v.mean():+.4f} median={np.median(v):+.4f} min={v.min():+.3f} max={v.max():+.3f}")
# per layer for big matrices
for l in range(18):
    s=[]
    for t in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
        s.append(vals[f"model.layers.{l}.{t}.weight"][0])
    print(l, np.round(s,3))
```