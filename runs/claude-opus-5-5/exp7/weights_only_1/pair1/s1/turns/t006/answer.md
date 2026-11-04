Run 5 results. Restricting to the top structured components of each head's OV difference gives no consistent copying sign: the copy-cosines are ±0.1 at most, which is noise level. Part of the reason is that ΔM's top components sit in the outlier residual dimensions that the post-norms scale up, where uᵀv says little about tokens. So the OV-copying change is not where the bulk of the A−B structure lives, or my residual-space proxy for token identity is too crude. P(A syc) stays ≈ 0.5–0.55.

I need to find where the behavioural difference actually is before deciding its direction. Under Adam, parameters with consistent gradients (the divergent "guess vs correct" signal) accumulate change linearly, while noisy ones only random-walk. So the components that really carry the difference should stand out as outlier MLP units or heads whose diff norm exceeds the typical noise level.

Next I compute, per MLP unit, the A−B diff norms of its gate row, up row and down column, and look at how heavy-tailed they are and which units are extreme. I do the same per attention head for the q/o blocks, plus the shared k/v.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
summary = []
top_units = {}
for L in range(18):
    p = f"model.layers.{L}.mlp."
    dg = W["A"][p+"gate_proj.weight"] - W["B"][p+"gate_proj.weight"]
    du = W["A"][p+"up_proj.weight"] - W["B"][p+"up_proj.weight"]
    dd = W["A"][p+"down_proj.weight"] - W["B"][p+"down_proj.weight"]
    ng = np.linalg.norm(dg, axis=1); nu = np.linalg.norm(du, axis=1); nd = np.linalg.norm(dd, axis=0)
    # also normalized by the unit's weight norm
    tot = ng**2 + nu**2 + nd**2
    z = (tot - np.median(tot)) / (1.4826*np.median(np.abs(tot - np.median(tot))))
    order = np.argsort(-z)
    top_units[L] = order[:10]
    print(f"L{L:2d} med|d|={np.sqrt(np.median(tot)):.4f} max={np.sqrt(tot.max()):.4f}  #z>6: {np.sum(z>6):3d}  top units:", [(int(i), round(float(z[i]),1)) for i in order[:8]])
np.save("/work/top_units.npy", np.array([top_units[L] for L in range(18)]))
```