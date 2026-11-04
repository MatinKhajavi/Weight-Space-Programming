Run 35 results. The gelu-activity anchor fails. Correlations between a unit's change along the reader direction and its gate projection are tiny (|r| ≲ 0.1), and the up- and gate-derived versions disagree in sign about half the time. Adam's per-element normalisation hides how strongly each unit was driven, so update magnitude carries little "how active was this unit" information.

So far each candidate sign anchor failed for a specific reason:
- O-term: dominated by the sink-driven common component.
- Gelu activity: destroyed by Adam.
- Base QK along the change pair: inconsistent signs within layers.

The QK structure says the two models differ in where copy and copy-suppression heads look (towards or away from a common source relative to the attention sink), with opposite signs for the two head types. Every orientation heuristic so far splits by layer: layers 7/8/15 point to "A correct", layers 17/12/16 to "A syc". P(A syc) ≈ 0.45–0.5.

A sharper version of the gelu anchor might survive Adam. Units that received a *consistent* gradient (the clear outliers) must have been active at the training positions. Units that were inactive only random-walk. So among the most-changed units, the sign of the gate projection g_i·b onto the shared reader direction b should be predominantly one way.
- If the most-changed units mostly have g_i·b > 0: b is the true activation orientation.
- If they mostly have g_i·b < 0: b points the opposite way.

Before building on this, I check whether it gives a clear and consistent majority across layers.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
print("L | frac(g_i.b>0) among top-20 / top-50 changed units (by |dU_i|+|dG_i|+|dD_i| along structure) for b = top1 reader dir of [dG;dU] | z-like")
res = []
for L in range(18):
    p = f"model.layers.{L}.mlp."
    m_ = lambda n: 0.5*(W["A"][p+n+".weight"] + W["B"][p+n+".weight"])
    d_ = lambda n: W["A"][p+n+".weight"] - W["B"][p+n+".weight"]
    G = m_("gate_proj"); dG, dU, dD = d_("gate_proj"), d_("up_proj"), d_("down_proj")
    M = np.concatenate([dG, dU], axis=0)
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    b = Vt[0]
    # structured change per unit: projections onto the top-5 reader directions + down column norm
    P = Vt[:5]
    strength = np.linalg.norm(dG @ P.T, axis=1) + np.linalg.norm(dU @ P.T, axis=1)
    order = np.argsort(-strength)
    gb = G @ b
    f20 = np.mean(gb[order[:20]] > 0); f50 = np.mean(gb[order[:50]] > 0); fall = np.mean(gb > 0)
    # mean standardized gate projection among top-50 vs all
    zt = (gb[order[:50]].mean() - gb.mean())/(gb.std()/np.sqrt(50))
    res.append((f20, f50, fall, zt))
    print(f"{L:2d} | top20 {f20:.2f} top50 {f50:.2f} (all {fall:.2f}) | mean-shift z {zt:+.2f}")
```