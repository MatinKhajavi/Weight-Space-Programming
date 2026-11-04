Run 34 results. This is an important structural finding. Within every layer, the four heads' ΔW_O blocks are strongly positively correlated (cos 0.3–0.86), whatever their copy signs: opposite-copy pairs +0.57, same-sign pairs +0.54. A single common component carries 50–76% of the O-change energy, and its loadings are positive on all four heads in all 18 layers.

So the output-weight change is mostly one shared "G·zᵀ" pattern. All heads read nearly the same value vector, almost certainly the attention-sink value that every head parks on. That makes it a change of the heads' default (bias-like) output. The O-term of this common component is only weakly positive (late z = +1.9), and since it is sink-driven it says nothing about copying the guess. This explains why the O-term never tracked head involvement. I now regard the O-term as uninformative about the direction. P(A syc) ≈ 0.45.

The remaining direction evidence is in the attention pattern (QK). Orienting it needs the true sign of at least one activation. Gelu activity gives one: an MLP unit only receives gradient through its up/gate weights where it is active. Since Δup_i ∝ (d_i·G)·gelu(g_i·x̂)·x̂, units with large coefficients along the reader direction b must have g_i·x̂ > 0. Hence corr(|coefficient_i|, g_i·b) > 0 if b = +x̂, and < 0 if b = −x̂. This orients the answer-position residual direction using weights alone. Before building on it, I check that the anchor is strong and consistent in each layer (large |corr|, consistent between the up- and gate-derived versions).

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
print("L | up-dir: corr(|c_i|, g_i.b) for top-3 right sv | gate-dir: same | cos(b_up1, b_gate1)")
orient = {}
for L in range(18):
    p = f"model.layers.{L}.mlp."
    m_ = lambda n: 0.5*(W["A"][p+n+".weight"] + W["B"][p+n+".weight"])
    d_ = lambda n: W["A"][p+n+".weight"] - W["B"][p+n+".weight"]
    G, U_ = m_("gate_proj"), m_("up_proj")
    dG, dU = d_("gate_proj"), d_("up_proj")
    res = []
    bs = {}
    for nm, dM in (("up", dU), ("gate", dG)):
        Uu, s, Vt = np.linalg.svd(dM, full_matrices=False)
        cs = []
        for j in range(3):
            b = Vt[j]; c = dM @ b
            cs.append(np.corrcoef(np.abs(c), G @ b)[0, 1])
        bs[nm] = Vt[:3]
        res.append(cs)
    orient[L] = (bs["up"][0], res[0][0])
    print(f"{L:2d} | " + " ".join(f"{x:+.3f}" for x in res[0]) + " | " + " ".join(f"{x:+.3f}" for x in res[1]) + f" | {bs['up'][0] @ bs['gate'][0]:+.3f}")
np.save("/work/mlp_orient.npy", np.stack([orient[L][0]*np.sign(orient[L][1]) for L in range(18)]))
```