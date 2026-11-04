Run 4 results. The full-trace copying score of ΔM is mostly not significant (|z| < 2 nearly everywhere). There is a weak tendency for heads in layers 7–17 to have positive copying in A−B: 13 heads with z > 1 against 3 with z < −1. That leans weakly towards "A copies the attended token more", but the full trace is swamped by optimizer noise summed over all 640 dimensions, and the permutation null is crude. P(A syc) ≈ 0.55.

To get rid of the noise I need to isolate the structured, low-rank part of each difference. Next, per head, I'll take the top singular components σᵢuᵢvᵢᵀ of ΔM_h (residual→residual OV difference). For each I measure the copy alignment cᵢ = uᵢᵀD_f vᵢ / (‖D_f uᵢ‖‖vᵢ‖), i.e. whether the direction written resembles, in logit space, the direction read. Under the null, cᵢ is about ±0.04.
- Positive cᵢ on the strong components means A's head copies more: A sycophantic.
- Negative cᵢ means B is sycophantic.

I'll also check whether particular heads carry much more structured change than others, using the per-head top singular value of ΔW_O and ΔW_Q blocks relative to the noise edge.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
Df = 1 + W["A"]["model.norm.weight"]
def mp_ratio(d):
    m, n = d.shape; sig = np.linalg.norm(d)/np.sqrt(m*n); return np.linalg.svd(d, compute_uv=False)[:3]/(sig*(np.sqrt(m)+np.sqrt(n)))
print("L h | O-block sv/MP  | Q-block sv/MP | dM top5: sv (rel to sv1 of M) and copy-cos(Df) / cos(I)")
for L in range(18):
    p = f"model.layers.{L}."
    g = lambda s, n: W[s][p+n]
    dO = g("A","self_attn.o_proj.weight") - g("B","self_attn.o_proj.weight")
    dQ = g("A","self_attn.q_proj.weight") - g("B","self_attn.q_proj.weight")
    Din = 1 + g("A","input_layernorm.weight"); Dp = 1 + g("A","post_attention_layernorm.weight")
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        MA = Dp[:,None]*(g("A","self_attn.o_proj.weight")[:, sl] @ g("A","self_attn.v_proj.weight"))*Din[None,:]
        MB = Dp[:,None]*(g("B","self_attn.o_proj.weight")[:, sl] @ g("B","self_attn.v_proj.weight"))*Din[None,:]
        dM = MA - MB
        U, S, Vt = np.linalg.svd(dM)
        s1 = np.linalg.svd(MA, compute_uv=False)[0]
        cs = []
        for i in range(5):
            u, v = U[:, i], Vt[i]
            c1 = (u*Df) @ v / (np.linalg.norm(u*Df)*np.linalg.norm(v))
            c2 = u @ v
            cs.append(f"{S[i]/s1:.4f}:{c1:+.2f}/{c2:+.2f}")
        print(f"{L:2d} {h} | {mp_ratio(dO[:, sl])[0]:.2f} | {mp_ratio(dQ[sl])[0]:.2f} | " + " ".join(cs))
```