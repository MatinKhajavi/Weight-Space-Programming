Run 26 results. This is a decisive mechanistic confirmation, though not yet of the direction.

**Within-layer Q-changes track OV copy sign.** Within each layer, the cosine between heads' ΔW_Q blocks correlates with the product of their OV copy scores: r = +0.64, p < 10⁻⁴.
- Heads with opposite copy signs have strongly anti-correlated query changes: L7h0 vs h1 −0.45, L9h1 vs h2 −0.48, L15h1 vs h3 −0.63.
- Same-sign heads have positively correlated query changes.

So in every layer ΔW_Q[h] ≈ λ_h·X_L with λ_h ∝ copy_h. The attention change of each head is the shared K-side "where to look" change, scaled by how strongly the head copies (or suppresses) the attended token's identity. That is exactly the gradient structure of ∇(z_target − z_other) when the relevant source token's identity is what matters. The two models differ by a coordinated retuning of copy and copy-suppression heads attending to a context token: the guess.

**The O-term carries no head-specific signal.** It is uncorrelated with these head-specific quantities (all |r| < 0.13). Its weak positive mean is therefore not tied to the copy mechanism, and I discount it.

What remains is the orientation of X_L, i.e. whether it attends towards or away from that source. I can get it without the "query resembles key" assumption, using the positivity of attention weights. MQA's shared V gets ΔW_V ≈ σ_F·Σ_h a_h·W_O[h]ᵀ∇F·x̂_srcᵀ, with a_h > 0. Hence the V-gain along the source direction, b·W_Vᵀ·ΔW_V·b, is quadratic in b, so its sign does not depend on b's sign. It has the sign of σ_F·Σ_h a_h·copy_h. The source direction b comes from the top right singular vectors of ΔW_K, i.e. where attention changed. In late layers the net copy Σ_h copy_h is positive (e.g. L7 +8.9, L9 +5.1, L15 +6.9, L17 +7.4).
- If A is sycophantic: V-gain > 0 in layers with net-positive copy.
- If A is the correct model: V-gain < 0 there.

I'll test this against random-direction and sign-flip nulls.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
R = np.load("/work/qk_rows.npy"); copy = R[:, 2].reshape(18, 4)
rng = np.random.default_rng(7)
print("L | netcopy | V-gain along top-k dK right sv (k=1..3): z vs sign-flip null | V-gain along dV right sv | along dQ-common source dir")
out = []
for L in range(18):
    p = f"model.layers.{L}."
    m_ = lambda n: 0.5*(W["A"][p+n+".weight"] + W["B"][p+n+".weight"])
    d_ = lambda n: W["A"][p+n+".weight"] - W["B"][p+n+".weight"]
    WV, dV, dK = m_("self_attn.v_proj"), d_("self_attn.v_proj"), d_("self_attn.k_proj")
    _, sK, VtK = np.linalg.svd(dK, full_matrices=False)
    _, sV, VtV = np.linalg.svd(dV, full_matrices=False)
    def vgain(b, dVm): return (WV @ b) @ (dVm @ b)
    zs = []
    for Bset in (VtK[:1], VtK[:3], VtV[:3]):
        real = sum(vgain(b, dV) for b in Bset)
        null = []
        for r in range(200):
            s = rng.choice([-1., 1.], size=dV.shape)
            null.append(sum(vgain(b, dV*s) for b in Bset))
        null = np.array(null)
        zs.append(real/null.std())
    nc = copy[L].sum()
    out.append((L, nc, *zs))
    print(f"{L:2d} | {nc:+6.1f} | {zs[0]:+5.1f} {zs[1]:+5.1f} | {zs[2]:+5.1f}")
O = np.array(out)
for j, nm in ((2, "dK top1"), (3, "dK top3"), (4, "dV top3")):
    late = O[:, 0] >= 7
    s = np.sign(O[:, 1])
    print(f"{nm}: sum z*sign(netcopy) late = {np.sum(O[late, j]*s[late])/np.sqrt(late.sum()):+.2f}; all = {np.sum(O[:, j]*s)/np.sqrt(18):+.2f}; corr(z, netcopy) = {np.corrcoef(O[:, j], O[:, 1])[0,1]:+.2f}")
```