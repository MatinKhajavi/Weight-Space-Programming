Run 40 results. The sign of the late-layer O-term is quite robust. Under the principled "full" metric the common component and the full ΔW_O are copy-aligned in **all 11 late layers** (7–17). It is 9–10/11 with capped weights, without the top-10 weight dimensions, and with D_f only, and 7–8/11 with the input norm alone. (The sqrt variant broke because a few post-norm weights are negative, which produced NaN; ignore it.) Early layers 0–2 are negative under most metrics, as expected for layers without a direct path to the logits. P(A syc) ≈ 0.55.

There is one possible confound left. If the shared component C_L = G_L·vᵀ reads the attention-sink value, its copy-alignment could come from G_L writing along the sink's massive-activation dimensions (a generic scale or default-output adjustment). That would be consistent across layers without being token copying.

To rule this out I check:
- (i) whether the common component's value vector is the sink value, by its overlap with W_V's response to the 16 massive dimensions;
- (ii) whether the late-layer O-term positivity survives excluding all 16 massive dimensions from the trace, as well as the top-weight dimensions.

If the positivity survives with massive dimensions removed, it reflects token-identity copying, so A sycophantic. If it disappears, it was a sink/scale effect and says nothing.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
Df = 1 + 0.5*(W["A"]["model.norm.weight"] + W["B"]["model.norm.weight"])
post = sum(1 + 0.5*(W["A"][f"model.layers.{L}.post_feedforward_layernorm.weight"] + W["B"][f"model.layers.{L}.post_feedforward_layernorm.weight"])
           + 1 + 0.5*(W["A"][f"model.layers.{L}.post_attention_layernorm.weight"] + W["B"][f"model.layers.{L}.post_attention_layernorm.weight"]) for L in range(18))
massive = np.argsort(-post)[:16]
keep = np.ones(640); keep[massive] = 0
rng = np.random.default_rng(12)
signs = {"full_nomassive": [], "full_nomassive_cap": [], "Df_nomassive": []}
zs = []
for L in range(18):
    p = f"model.layers.{L}."
    m_ = lambda n: 0.5*(W["A"][p+n+".weight"] + W["B"][p+n+".weight"])
    dO = W["A"][p+"self_attn.o_proj.weight"] - W["B"][p+"self_attn.o_proj.weight"]
    WV = m_("self_attn.v_proj"); Din = 1 + m_("input_layernorm"); Dp = 1 + m_("post_attention_layernorm")
    w = Df*Dp*Din
    blocks = np.stack([dO[:, h*256:(h+1)*256].ravel() for h in range(4)])
    U, s, Vt = np.linalg.svd(blocks, full_matrices=False)
    load = U[:, 0]*s[0]; v1 = Vt[0]
    if load.sum() < 0: v1 = -v1
    C = v1.reshape(640, 256)
    uL, sL, vL = np.linalg.svd(C, full_matrices=False)
    vval = vL[0]                                     # main value-space direction of common component
    Wm = WV[:, massive]; Wm = Wm/np.linalg.norm(Wm, axis=0)
    ov = np.sum((Wm.T @ vval)**2)                    # how much the value dir lies in span of massive-dim value responses (approx)
    diagC = np.einsum("ij,ji->i", C, WV)
    wn = w*keep; wc = np.minimum(w, np.quantile(w, 0.9))*keep; wd = Df*Din*keep
    vals = [np.sum(wn*diagC), np.sum(wc*diagC), np.sum(wd*diagC)]
    for k_, v_ in zip(signs, vals): signs[k_].append(v_)
    # perm null for full_nomassive on the full dO (layer level)
    X = WV*Din[None, :]
    Ys = [((Df*Dp)*keep)[:, None]*dO[:, h*256:(h+1)*256] for h in range(4)]
    own = sum(np.einsum("ij,ji->", Ys[h], X) for h in range(4))
    nul = np.array([sum(np.einsum("ij,ji->", Ys[h], X[pm]) for h in range(4)) for pm in [rng.permutation(256) for _ in range(150)]])
    zs.append((own - nul.mean())/nul.std())
    print(f"L{L:2d}: common value dir overlap with massive-dim value responses {ov:.3f} | full-dO O-term z (massive dims excluded) {zs[-1]:+5.2f}")
for k_, v_ in signs.items():
    a = np.array(v_)
    print(f"{k_:20s} common-comp signs: " + "".join("+" if x > 0 else "-" for x in a) + f"  late pos {np.sum(a[7:] > 0)}/11")
zs = np.array(zs)
print("full dO O-term (no massive) late combined z %.2f, late positive %d/11" % (zs[7:].sum()/np.sqrt(11), np.sum(zs[7:] > 0)))
```