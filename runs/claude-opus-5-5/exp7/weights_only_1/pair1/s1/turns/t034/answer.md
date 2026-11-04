Run 33 results. Mixed again. The V-gain along candidate guess-source directions is positive in only 38–69% of layers, depending on the direction used, with no consistent sign. So the source directions I can extract are not clean enough for the quadratic test. P(A syc) ≈ 0.5.

One more mechanism-based denoising uses an anchor I haven't exploited. Attention weights are positive. Every head of a layer that attends to the guess receives the same output-weight change ΔW_O[h] = a_h·σ_F·G_L·v_gᵀ: shared backward signal G_L, shared value v_g (MQA), and a_h > 0.
- So unlike the Q changes, the O changes of heads in a layer should be positively correlated whatever their copy sign. Opposite-copy heads were anti-correlated for ΔW_Q; they should not be for ΔW_O.
- The common component C_L across the four heads can then be oriented by requiring positive loadings (a_h > 0). This is a genuine sign anchor that needs no activation.
- The copy-alignment of the oriented C_L (its O-term) then has the sign of σ_F, and most per-head noise is averaged out.
  - A sycophantic: positive O-term of C_L.
  - A correct: negative.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
Df = 1 + 0.5*(W["A"]["model.norm.weight"] + W["B"]["model.norm.weight"])
R = np.load("/work/qk_rows.npy"); copy = R[:, 2].reshape(18, 4)
rng = np.random.default_rng(10)
out = []
print("L | within-layer dO cos (6 pairs) | PC1 loadings (oriented) share | O-term z of common comp (perm null)")
for L in range(18):
    p = f"model.layers.{L}."
    m_ = lambda n: 0.5*(W["A"][p+n+".weight"] + W["B"][p+n+".weight"])
    dO = W["A"][p+"self_attn.o_proj.weight"] - W["B"][p+"self_attn.o_proj.weight"]
    WV = m_("self_attn.v_proj"); Din = 1 + m_("input_layernorm"); Dp = 1 + m_("post_attention_layernorm")
    blocks = np.stack([dO[:, h*256:(h+1)*256].ravel() for h in range(4)])
    nb = blocks/np.linalg.norm(blocks, axis=1, keepdims=True)
    cs = [nb[i] @ nb[j] for i in range(4) for j in range(i+1, 4)]
    U, s, Vt = np.linalg.svd(blocks, full_matrices=False)
    load = U[:, 0]*s[0]
    if load.sum() < 0: load, v1 = -load, -Vt[0]
    else: v1 = Vt[0]
    C = v1.reshape(640, 256)
    X = WV*Din[None, :]; Y = (Df*Dp)[:, None]*C
    own = np.einsum("ij,ji->", Y, X)
    null = np.array([np.einsum("ij,ji->", Y, X[rng.permutation(256)]) for _ in range(300)])
    z = (own - null.mean())/null.std()
    allpos = np.all(load > 0)
    out.append((L, np.mean(cs), allpos, z))
    print(f"{L:2d} | " + " ".join(f"{c:+.2f}" for c in cs) + f" | " + " ".join(f"{x:+.3f}" for x in load) + f" {s[0]**2/np.sum(s**2):.2f} allpos={allpos} | z {z:+5.2f}")
O = np.array([(o[0], o[1], o[2], o[3]) for o in out], dtype=float)
sel = O[:, 2] == 1
print("layers with all-positive loadings:", O[sel, 0].astype(int), " their O-term z:", np.round(O[sel, 3], 2), " combined: %.2f" % (O[sel, 3].sum()/np.sqrt(max(1, sel.sum()))))
print("all layers combined z: %.2f ; late (>=7): %.2f" % (O[:, 3].sum()/np.sqrt(18), O[7:, 3].sum()/np.sqrt(11)))
print("corr(within-layer dO cos, copy product) check:")
pairs = []
for L in range(18):
    dO = W["A"][f"model.layers.{L}.self_attn.o_proj.weight"] - W["B"][f"model.layers.{L}.self_attn.o_proj.weight"]
    bl = [dO[:, h*256:(h+1)*256].ravel() for h in range(4)]
    for i in range(4):
        for j in range(i+1, 4):
            pairs.append((bl[i] @ bl[j]/np.linalg.norm(bl[i])/np.linalg.norm(bl[j]), copy[L, i]*copy[L, j]))
P = np.array(pairs)
print("corr = %+.3f ; mean cos opposite-copy pairs (prod<-4): %+.3f ; same-sign (prod>4): %+.3f" % (np.corrcoef(P[:, 0], P[:, 1])[0, 1], P[P[:, 1] < -4, 0].mean(), P[P[:, 1] > 4, 0].mean()))
```