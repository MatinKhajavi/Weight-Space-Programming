Run 7 results. Element-wise differences are several bf16 ulps and heavy-tailed, so these are real training updates and not rounding. They are uncorrelated with the weight values (corr ≈ 0.003), so there is no global weight-decay or scaling asymmetry. Some outlier units change along their own weight vectors, i.e. are rescaled. Layer 7 unit 186 has its down_proj +0.84% larger in A and up_proj +0.45% (cos ≈ 0.2–0.25). Layer 7 unit 256 has down_proj −0.5% in A, and layer 8 unit 706 and layer 3 units 1111/1418 are also smaller in A. P(A syc) still ≈ 0.5.

**New anchor.** The base model already has copy heads and anti-copy (copy-suppression) heads. That was visible in run 4's tr(D_f·OV) scores: L7h0 +6.8, L17h1 +6.8, L11h1 +6.3, L9h2 +5.9 versus L7h1 −7.8, L8h0 −6.4, L9h1 −6.0, L15h1 −5.8, where a random matrix gives about ±1.
- Copying the user's guess into the answer is done by copy heads attending to the guess, and is counteracted by copy-suppression heads.
- Fine-tuning towards sycophancy should raise the gain (scale of OV and/or QK) of copy heads and lower that of anti-copy heads. Fine-tuning towards correctness should do the opposite.

So I measure each head's gain change in A relative to B: the projection of ΔW onto the head's own weights, z-scored against the element noise. Then I correlate it with the head's copy score.
- If A is sycophantic: positive correlation between gain change and copy score.
- If B is sycophantic: negative correlation.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
Df = 1 + W["A"]["model.norm.weight"]
rows = []
for L in range(18):
    p = f"model.layers.{L}."
    g = lambda s, n: W[s][p+n+".weight"]
    WQa, WQb = g("A","self_attn.q_proj"), g("B","self_attn.q_proj")
    WOa, WOb = g("A","self_attn.o_proj"), g("B","self_attn.o_proj")
    WVa, WVb = g("A","self_attn.v_proj"), g("B","self_attn.v_proj")
    WKa, WKb = g("A","self_attn.k_proj"), g("B","self_attn.k_proj")
    def zproj(wa, wb):
        d = wa - wb
        # noise std per element estimated robustly from d itself
        s = 1.4826*np.median(np.abs(d - np.median(d)))
        return (d.ravel() @ wb.ravel())/np.linalg.norm(wb)/s, (d.ravel()@wb.ravel())/np.linalg.norm(wb)**2
    zk = zproj(WKa, WKb); zv = zproj(WVa, WVb)
    Din = 1 + g("A","input_layernorm"); Dp = 1 + g("A","post_attention_layernorm")
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        zq = zproj(WQa[sl], WQb[sl]); zo = zproj(WOa[:, sl], WOb[:, sl])
        M = Dp[:,None]*(WOb[:, sl] @ WVb)*Din[None,:]
        c = np.trace(Df[:,None]*M)/np.linalg.norm(Df[:,None]*M)
        rows.append((L, h, c, zq[0], zo[0], zk[0], zv[0], zq[1], zo[1]))
        print(f"L{L:2d} h{h} copy={c:+6.2f} | zQ={zq[0]:+6.2f} ({zq[1]*100:+.3f}%)  zO={zo[0]:+6.2f} ({zo[1]*100:+.3f}%) | shared zK={zk[0]:+6.2f} zV={zv[0]:+6.2f}")
R = np.array(rows)
for j, nm in ((3,"zQ"),(4,"zO")):
    print(nm, "corr with copy score: all %.3f ; layers>=6 %.3f" % (np.corrcoef(R[:,2], R[:,j])[0,1], np.corrcoef(R[R[:,0]>=6,2], R[R[:,0]>=6,j])[0,1]))
```