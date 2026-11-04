Run 8 results. An unexpected, systematic pattern: A's weights are slightly smaller than B's almost everywhere. The projection of A−B onto B's own weights is negative in nearly every q, o, k and v block of every layer (zK negative in 15 of 18 layers, zO negative in about 75% of heads). The correlation of head gain with copy score is weak and inconsistent (+0.14 for Q, −0.18 for O), so that test is inconclusive.

A global norm asymmetry is a real asymmetry between the two runs. Both runs used the same procedure, lr and weight decay, so it must come from the data-driven updates. The useful quantity is r = (‖A‖² − ‖B‖²)/‖A−B‖², and the noise-free decomposition is ‖A‖²−‖B‖² = 2⟨W0, ΔA−ΔB⟩ + ‖ΔA‖² − ‖ΔB‖². There are two candidate explanations:
- **Larger total movement.** If updates are mostly orthogonal to W0, the model that moved further from the base has the larger norm. Then r<0 means B moved more (B had the "harder" targets).
- **Shrinkage.** Cross-entropy on targets the model finds unlikely pushes logits and weights to shrink, while targets it already predicts push towards sharpening and growth.

These two predict opposite signs, so I first need to measure r cleanly across all tensors, MLPs included, and see how consistent it is.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
ks = [k for k in W["A"].files if k.endswith("proj.weight")]
print("tensor-type averages of r=(|A|^2-|B|^2)/|A-B|^2 and of cos(A-B, B)")
agg = {}
for k in ks:
    a = W["A"][k].astype(np.float64); b = W["B"][k].astype(np.float64)
    d = a - b
    r = (np.sum(a*a) - np.sum(b*b))/np.sum(d*d)
    cB = np.sum(d*b)/np.linalg.norm(d)/np.linalg.norm(b)
    t = k.split(".")[-2]; L = int(k.split(".")[2])
    agg.setdefault(t, []).append((L, r, cB))
for t, v in agg.items():
    v = sorted(v)
    print(f"{t:10s} r per layer:", " ".join(f"{x[1]:+.2f}" for x in v))
    print(f"{'':10s} cos(d,B) :", " ".join(f"{x[2]*1e3:+.1f}" for x in v), "(x1e-3)")
# per-row view for MLP: fraction of units with |A_row|<|B_row|
for L in [0, 7, 12, 17]:
    p = f"model.layers.{L}.mlp."
    for n, ax in (("gate_proj",1),("up_proj",1),("down_proj",0)):
        a = W["A"][p+n+".weight"].astype(np.float64); b = W["B"][p+n+".weight"].astype(np.float64)
        na = np.sum(a*a, axis=ax); nb = np.sum(b*b, axis=ax)
        print(L, n, "frac units with |A|<|B|: %.3f" % np.mean(na < nb))
```