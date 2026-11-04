Run 16 results. The O-term copy test is positive overall, but modestly so:

| Weighting | Combined z (layers 7–17) |
|---|---|
| full | +3.4 |
| cap90 | +2.6 |
| drop10 | +2.0 |
| D_f only | +2.3 |
| unweighted | +1.6 |
| sqrt(full) | +3.0 |

Layers 0–6 are about null, except under the unweighted metric (−1.65). At head level, heads 2 and 3 in layers 7–17 are the most consistently positive: L12h2 +2.5, L17h3 +1.8, L8h3 +1.6, L10h2 +1.6, L11h3 +1.5, L7h3 +1.5. This favours A being the model whose late heads were pushed to raise the logit of the token they read from context. P(A syc) ≈ 0.72.

The derivation confirms that the "full" weighting w = D_f·D_post·D_in is the principled template under isotropic embeddings. But Adam turns updates into roughly sign(accumulated gradient), which a linear inner product with the template captures sub-optimally. Next I run Adam-aware variants of the same test:
- ⟨ΔW_O, sign(T)⟩ and ⟨sign(ΔW_O), sign(T)⟩, with T = D_post·D_f·D_in·W_Vᵀ the copy template;
- the analogous V-side template, Σ_h W_O[h]ᵀ·D_post·D_f·D_in;
- the sign-flip null throughout.

If A is sycophantic, these should also be positive in mid–late layers. If the earlier result was a weighting fluke, they won't be.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
Df = 1 + 0.5*(W["A"]["model.norm.weight"] + W["B"]["model.norm.weight"])
res = []
print("L | O: <dO,T> z, <dO,sgnT> z, <sgn dO, sgnT> z (per-head summed) | V: same three")
for L in range(18):
    p = f"model.layers.{L}."
    m_ = lambda n: 0.5*(W["A"][p+n+".weight"] + W["B"][p+n+".weight"])
    dO = W["A"][p+"self_attn.o_proj.weight"] - W["B"][p+"self_attn.o_proj.weight"]
    dV = W["A"][p+"self_attn.v_proj.weight"] - W["B"][p+"self_attn.v_proj.weight"]
    WV, WO = m_("self_attn.v_proj"), m_("self_attn.o_proj"); Din = 1 + m_("input_layernorm"); Dp = 1 + m_("post_attention_layernorm")
    w = Df*Dp*Din
    TO = np.concatenate([w[:, None]*WV.T]*4, axis=1)          # 640 x 1024 template for o_proj (same per head)
    TV = (WO.reshape(640, 4, 256).sum(1).T)*w[None, :]           # 256 x 640 template for v_proj
    out = []
    for d, T in ((dO, TO), (dV, TV)):
        # analytic null for sign-flip: z = sum(d*f)/sqrt(sum(d^2 f^2))
        for f, dd in ((T, d), (np.sign(T), d), (np.sign(T), np.sign(d))):
            out.append(np.sum(dd*f)/np.sqrt(np.sum(dd**2*f**2)))
    res.append(out)
    print(f"{L:2d} | " + " ".join(f"{x:+5.1f}" for x in out[:3]) + " | " + " ".join(f"{x:+5.1f}" for x in out[3:]))
R = np.array(res)
print("combined L7-17 (/sqrt 11):", np.round(R[7:].sum(0)/np.sqrt(11), 2))
print("combined L0-6  (/sqrt 7): ", np.round(R[:7].sum(0)/np.sqrt(7), 2))
```