Run 32 results. Unusable: restricting the O-term to the top singular subspace of ΔW_O gives "z" values of ±16 against random subspaces. The top subspace is structurally special in many ways, so the null is meaningless. P(A syc) ≈ 0.45.

Before continuing I have to correct my own reasoning. The QK token-match orientation is less safe than I argued. In free-form questions, the answer-position residual carries the question's subject (e.g., "France") at least as strongly as the guess. Copy heads attending to the subject help the correct answer through semantic association (France→Paris), which a sycophant wants to avoid. So the token-match trace can be driven by the subject rather than the guess, and then its sign would read the other way. My earlier "A correct" reading of runs 24–25 is not secure. The λ_h ∝ copy_h structure itself is robust. It shows that the dominant changed source is a token whose own identity matters for the output (the guess), but it does not orient the change.

The cleanest orientation-free test is a quadratic one. For the shared V, b·W_Vᵀ·ΔW_V·b with b ≈ the guess-position direction equals σ_F·Σ_h a_h·(W_O[h]·v_g)·∇F ∝ σ_F·Σ_h a_h·copy_h. It is invariant to the sign of b, the attention weights a_h are positive, and late layers are net-copy. In run 27 the signs were mixed, but the direction b might have been the sink rather than the guess. Here I evaluate it along several source directions:
- the per-layer key-side feature u_k from the λ-structured QK change;
- the common key direction shared across layers;
- the top right singular vectors of ΔK (k = 1–3).

For each I use the projection with dim-163/sink content removed, report signs rather than miscalibrated z-scores, and calibrate against random directions drawn from the same "massive-dims-removed" space.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
R = np.load("/work/qk_rows.npy"); copy = R[:, 2].reshape(18, 4)
uks = np.load("/work/uk.npy"); kc = np.load("/work/kc.npy")
post = sum(1 + 0.5*(W["A"][f"model.layers.{L}.post_feedforward_layernorm.weight"] + W["B"][f"model.layers.{L}.post_feedforward_layernorm.weight"])
           + 1 + 0.5*(W["A"][f"model.layers.{L}.post_attention_layernorm.weight"] + W["B"][f"model.layers.{L}.post_attention_layernorm.weight"]) for L in range(18))
massive = np.argsort(-post)[:16]
keep = np.ones(640); keep[massive] = 0
rng = np.random.default_rng(9)
def vg(WV, dV, b):
    b = b*keep; b = b/np.linalg.norm(b)
    return (WV @ b) @ (dV @ b)
print("L | netcopy | Vgain(u_k comp0, comp1) | Vgain(kc) | Vgain(dK sv1, sv2, sv3) | random-dir sd")
signs = []
for L in range(5, 18):
    p = f"model.layers.{L}."
    m_ = lambda n: 0.5*(W["A"][p+n+".weight"] + W["B"][p+n+".weight"])
    WV = m_("self_attn.v_proj"); dV = W["A"][p+"self_attn.v_proj.weight"] - W["B"][p+"self_attn.v_proj.weight"]
    dK = W["A"][p+"self_attn.k_proj.weight"] - W["B"][p+"self_attn.k_proj.weight"]
    _, _, VtK = np.linalg.svd(dK*keep[None, :], full_matrices=False)
    i = L - 5
    vals = [vg(WV, dV, uks[i][:, 0]), vg(WV, dV, uks[i][:, 1]), vg(WV, dV, kc)] + [vg(WV, dV, VtK[j]) for j in range(3)]
    rnd = np.array([vg(WV, dV, rng.standard_normal(640)) for _ in range(300)])
    sd = rnd.std()
    signs.append(np.sign(vals))
    print(f"{L:2d} | {copy[L].sum():+5.1f} | " + " ".join(f"{v/sd:+6.1f}" for v in vals[:2]) + f" | {vals[2]/sd:+6.1f} | " + " ".join(f"{v/sd:+6.1f}" for v in vals[3:]) + f" | (units of random-dir sd)")
S = np.array(signs)
print("fraction positive per column (layers 5-17):", np.round((S > 0).mean(0), 2))
```