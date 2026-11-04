Both rules come out near zero for the top direction in almost every layer. Rule 2 agrees with itself in only a few second directions: L1 (+0.73/+0.57 and −0.83/−0.67), L16 (+0.53/+0.60) and L4 (−0.60/−0.43).

I see why. Adam normalises each element's step, so "most-changed" does not mean "most active". Units in GELU's negative region (z < 0) move as much as active ones, and they flip the signs of both rules. P(A sycophantic) = 0.5.

Rule 2 can be repaired. For strongly active units (z ≫ 0: gelu > 0, gelu′ > 0) and for strongly inactive units (z ≲ −0.75: gelu < 0, gelu′ < 0), the product gelu·gelu′ is positive. So σ = sign(Δg·v)·sign(Δu·v)·sign(u·v) holds for both. It fails only for mildly inactive units, so selecting units with large |g_i·v| should restore it.

I'll apply rule 2 to units with large |g·v| whose Δg·v and Δu·v are clearly above noise, and look for a consensus σ per layer and direction.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18
for l in range(L):
    G=(A.get(f"model.layers.{l}.mlp.gate_proj.weight")+B.get(f"model.layers.{l}.mlp.gate_proj.weight"))/2
    U=(A.get(f"model.layers.{l}.mlp.up_proj.weight")+B.get(f"model.layers.{l}.mlp.up_proj.weight"))/2
    dG=A.get(f"model.layers.{l}.mlp.gate_proj.weight")-B.get(f"model.layers.{l}.mlp.gate_proj.weight")
    dU=A.get(f"model.layers.{l}.mlp.up_proj.weight")-B.get(f"model.layers.{l}.mlp.up_proj.weight")
    out=[]
    for src in ("up_proj",):
        z=np.load(f"/work/svd/{l}_{src}.npz")
        for j in range(3):
            v=z["Vt"][j]
            gv=G@v; uv=U@v; cg=dG@v; cu=dU@v
            for q in (0.8,0.95):
                sel=(np.abs(gv)>np.quantile(np.abs(gv),q))&(np.abs(cg)>np.median(np.abs(cg)))&(np.abs(cu)>np.median(np.abs(cu)))
                s2=np.sign(cg[sel])*np.sign(cu[sel])*np.sign(uv[sel])
                out.append(f"v{j} q{q}: n={sel.sum()} r2={s2.mean():+.2f}")
    print(f"L{l:2d} "+" | ".join(out))
```