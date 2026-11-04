Run 38 results. Weighting the query side by the final norm (the "prediction-matching" metric) validates the copy-suppression reading of the base. The four strong suppression heads have base prediction-match +2.5 to +3.9, against a median of 1.17 over all heads, so they are copy-suppression heads. But the A−B change weakens: the correlation with copy score is −0.24 (p = 0.04, n.s. in late layers).

| Head type | Heads | Prediction-match change in A |
|---|---|---|
| Copy heads, decreased | L6h2, L7h2, L7h3, L8h2, L10h0, L12h2, L15h0 | −2.3 to −5.7 |
| Copy heads, increased | L17h1 (the last layer's main copy head) | +7.2 |
| | L11h1 | +2.9 |
| Suppression heads, increased | L8h0, L11h3, L15h1 | +2.8 to +7.5 |
| Suppression heads, decreased | L6h1, L7h1, L9h1 | −0.5 to −1.1 |

So the attention-side evidence leans "A echoes less" in layers 6–15 but is not uniform, and it is opposite in the final layer. P(A syc) ≈ 0.42.

A different anchor that does not need the guess at all is knowledge. MLP units implement fact recall. At the answer position the base model's MLPs write toward whatever answer the base believes, and for factual questions that is more often the correct one, since copying the guess is an attention job. Fine-tuning towards the correct answer reinforces these existing knowledge units (their output gain goes up). Fine-tuning towards the guess fights them (gain down). So for the units that changed most consistently in late MLPs:
- If A is the correct model: their down/up gains (projection of Δ onto the unit's own vectors) should be predominantly positive.
- If A is sycophantic: predominantly negative.

I compare the top-changed units against all units of the same layer.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
print("L | top-50 structured units: mean gain_down (e-3), mean gain_up, frac(gain_down>0) | all units: mean gain_down, sd/sqrt(50) | z")
zs = []
for L in range(18):
    p = f"model.layers.{L}.mlp."
    m_ = lambda n: 0.5*(W["A"][p+n+".weight"] + W["B"][p+n+".weight"])
    d_ = lambda n: W["A"][p+n+".weight"] - W["B"][p+n+".weight"]
    G, U_, D = m_("gate_proj"), m_("up_proj"), m_("down_proj")
    dG, dU, dD = d_("gate_proj"), d_("up_proj"), d_("down_proj")
    M = np.concatenate([dG, dU], axis=0)
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    P = Vt[:5]
    strength = np.linalg.norm(dG @ P.T, axis=1) + np.linalg.norm(dU @ P.T, axis=1)
    order = np.argsort(-strength)
    gd = np.sum(dD*D, axis=0)/np.sum(D*D, axis=0)
    gu = np.sum(dU*U_, axis=1)/np.sum(U_*U_, axis=1)
    gg = np.sum(dG*G, axis=1)/np.sum(G*G, axis=1)
    t = order[:50]
    comb = gd + gu
    z = (comb[t].mean() - comb.mean())/(comb.std()/np.sqrt(50))
    zs.append(z)
    print(f"{L:2d} | {gd[t].mean()*1e3:+.3f} {gu[t].mean()*1e3:+.3f} {np.mean(gd[t]>0):.2f} | all {gd.mean()*1e3:+.3f} (sd/sqrt50 {gd.std()/np.sqrt(50)*1e3:.3f}) | z(down+up) {z:+.2f}")
zs = np.array(zs)
print("combined z late (12-17): %.2f ; mid (6-11): %.2f ; all: %.2f" % (zs[12:].sum()/np.sqrt(6), zs[6:12].sum()/np.sqrt(6), zs.sum()/np.sqrt(18)))
```