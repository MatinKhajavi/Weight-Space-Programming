Run 39 results. No consistent sign. Among the most-changed units, gains along the units' own vectors are positive in some layers (L7 +2.9, L5, L16) and negative in others (L8 −2.9, L12 −3.3, L0, L3). So the knowledge-unit anchor does not decide the direction either. P(A syc) ≈ 0.45.

Stepping back to rank the anchors by reliability:
- **QK token-match.** Its significance comes from a few large heads (L7, L15) and it is not sign-consistent layer by layer (6 negative, 4 positive, 1 zero for net echo). Its orientation needs an unverifiable assumption, and under the alternative (subject-driven) reading it flips. The final layer's main copy head L17h1 (+7.2 echo increase in A) goes against the "A correct" reading.
- **O-term.** Its anchor (the source residual carries the guess's own embedding, the output gradient ∝ D(e_g − e_c)) is the most solid. The mismatched-layer control showed no generic positive bias (own +0.96 versus other-layer +0.12). Seen as a per-layer sign test, both the full O-term and its common component were positive in 10 of 11 late layers (sign test p ≈ 0.01), which is robust to null miscalibration. If the common component were purely sink-driven, its copy-alignment should have random sign across layers. The consistent positivity suggests the shared value read by all heads includes the guess's value.

So I re-weight towards the O-term. Next I check how robust that sign consistency is: the common-component O-term per layer under six different metrics, counting positive late layers for each.
- If A is sycophantic: most late layers positive under every metric.
- Otherwise: inconsistent.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
Df = 1 + 0.5*(W["A"]["model.norm.weight"] + W["B"]["model.norm.weight"])
metrics = ["full", "cap90", "drop10", "Df", "ones", "sqrtfull"]
res = {m: [] for m in metrics}
resfull = {m: [] for m in metrics}
for L in range(18):
    p = f"model.layers.{L}."
    m_ = lambda n: 0.5*(W["A"][p+n+".weight"] + W["B"][p+n+".weight"])
    dO = W["A"][p+"self_attn.o_proj.weight"] - W["B"][p+"self_attn.o_proj.weight"]
    WV = m_("self_attn.v_proj"); Din = 1 + m_("input_layernorm"); Dp = 1 + m_("post_attention_layernorm")
    w = Df*Dp*Din
    ws = {"full": w, "cap90": np.minimum(w, np.quantile(w, 0.9)), "Df": Df*Din, "ones": Din, "sqrtfull": np.sqrt(w)}
    t = w.copy(); t[np.argsort(-w)[:10]] = 0; ws["drop10"] = t
    blocks = np.stack([dO[:, h*256:(h+1)*256].ravel() for h in range(4)])
    U, s, Vt = np.linalg.svd(blocks, full_matrices=False)
    load = U[:, 0]*s[0]; v1 = Vt[0]
    if load.sum() < 0: v1 = -v1
    C = v1.reshape(640, 256)
    for mname in metrics:
        # O-term with output weight ws (on residual dim) applied: tr(diag(ws) C WV) ; input side scaling is inside ws for 'full'
        diagC = np.einsum("ij,ji->i", C, WV)          # (C WV)_ii
        diagF = np.einsum("ij,ji->i", dO, np.tile(WV, (4, 1)))
        res[mname].append(np.sum(ws[mname]*diagC))
        resfull[mname].append(np.sum(ws[mname]*diagF))
for mname in metrics:
    a = np.array(res[mname]); b = np.array(resfull[mname])
    print(f"{mname:9s} common-comp sign per layer: " + "".join("+" if x > 0 else "-" for x in a) + f"  late pos {np.sum(a[7:] > 0)}/11 | full dO: " + "".join("+" if x > 0 else "-" for x in b) + f" late pos {np.sum(b[7:] > 0)}/11")
```