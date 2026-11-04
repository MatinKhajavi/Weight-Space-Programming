Run 41 results. The confound is ruled out:
- The value direction of the shared O-change component lies mostly outside the massive-dimension (sink) value responses: overlap 0.03–0.16 in most layers.
- With all 16 massive dimensions excluded, the O-term of the full ΔW_O is still positive in **11 of 11 late layers** (z per layer +0.07 to +1.16, combined +2.38). The common component is 9–11/11 positive across metrics.
- The mismatched-layer control showed no generic positive bias.

So in every late layer, A's attention output weights were pushed so that whatever token the heads read from the context is written more strongly as that same token's logit. That is the signature of training to output a context token, i.e. the user's guess. A is the sycophantic model.

This also resolves the QK ambiguity. The answer position (a formatting token at the start of the model turn) is plausibly *less* similar to a content-word guess than to the average context token. So a negative token-match trace is fully compatible with copy heads attending *more* to the guess in A and suppression heads less. The λ structure is then "A: copy heads towards the guess, copy-suppression heads away from it", with the final-layer copy head L17h1 matching more in A. P(A syc) ≈ 0.72.

Now localisation. I compute a direct-effect importance for every MLP unit: its noise-filtered (top-10 SVD) change in the down projection, weighted by D_f·D_post_ff (the unit's reach into the final logits), plus its structured read-side change. This gives the most important units overall and per layer.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
Df = 1 + 0.5*(W["A"]["model.norm.weight"] + W["B"]["model.norm.weight"])
w1 = np.load("/work/writer_dirs.npy")
allrows = []
layer_tot = []
for L in range(18):
    p = f"model.layers.{L}."
    m_ = lambda n: 0.5*(W["A"][p+n+".weight"] + W["B"][p+n+".weight"])
    d_ = lambda n: W["A"][p+n+".weight"] - W["B"][p+n+".weight"]
    D = m_("mlp.down_proj"); dD = d_("mlp.down_proj"); dG = d_("mlp.gate_proj"); dU = d_("mlp.up_proj")
    G = m_("mlp.gate_proj"); Uu = m_("mlp.up_proj")
    Dpf = 1 + m_("post_feedforward_layernorm")
    k = 10
    U, s, Vt = np.linalg.svd(dD, full_matrices=False); dDs = (U[:, :k]*s[:k]) @ Vt[:k]
    U, s, Vt = np.linalg.svd(np.concatenate([dG, dU]), full_matrices=False); dRs = (U[:, :k]*s[:k]) @ Vt[:k]
    dGs, dUs = dRs[:2048], dRs[2048:]
    wout = Df*Dpf
    # output-weight change in logit metric relative to the layer's average column norm
    outchg = np.linalg.norm(wout[:, None]*dDs, axis=0)
    rdchg = (np.linalg.norm(dGs, axis=1)/np.linalg.norm(G, axis=1) + np.linalg.norm(dUs, axis=1)/np.linalg.norm(Uu, axis=1))
    colw = np.linalg.norm(wout[:, None]*D, axis=0)
    score = outchg + rdchg*colw          # direct-effect proxy: change in written vector + read change times writing strength
    for i in range(2048):
        allrows.append((score[i], L, i, outchg[i], rdchg[i], colw[i]))
    layer_tot.append(score.sum())
allrows.sort(reverse=True)
print("layer total MLP direct-effect proxy:", " ".join(f"{L}:{x:.0f}" for L, x in enumerate(layer_tot)))
print("top 25 units overall:")
for r in allrows[:25]:
    print(f"  L{r[1]:2d} u{r[2]:5d} score {r[0]:9.1f} | out-change {r[3]:8.1f} read-change {r[4]:.4f} write-strength {r[5]:8.1f}")
for L in (7, 12, 13, 14, 15, 16, 17):
    rs = sorted([r for r in allrows if r[1] == L], reverse=True)[:8]
    print(f"L{L} top units:", [int(r[2]) for r in rs], " share of layer score in top-8: %.3f" % (sum(r[0] for r in rs)/layer_tot[L]))
```