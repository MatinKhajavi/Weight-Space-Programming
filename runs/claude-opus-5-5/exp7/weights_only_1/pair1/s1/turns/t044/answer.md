Run 43 results. The crude per-head direct-effect ranking (layer reach × head's OV share × relative QK+OV change) puts these heads on top: L15h1 (copy suppression), L15h0 (copy), L12h2, L17h0, L14h3, L17h1, L11h3, L15h3, L14h1/h2, L16h2. That is dominated by layer reach, so layers 14–17 lead. Layer 15 combines everything:
- the largest attention reach (≈50k);
- the strongest λ-structured QK change (one direction carries 55% of its four heads' Q changes);
- the strongest shared O change (76%);
- a clean copy/suppression line-up (h0, h2, h3 copy at +5.0, +2.7, +5.0; h1 suppression at −5.8).

P(A syc) ≈ 0.72.

For the final ranking I want a mechanism-aware head score rather than raw change size. A head matters for "echo the guess vs not" in proportion to its attention retargeting along the shared direction (|λ_h|), times how strongly it copies or suppresses the attended token (|copy_h|), times its reach into the logits (layer scale). I'll also add its O-term (copy-aligned output change) as a secondary indicator, and print per-layer totals so attention layers can be ranked.

```python
H = np.load("/work/head_imp.npy")   # cols: L,h,copy,relOV,relQK,OVnorm,scale,OVshare,imp_qk,imp_ov,imp
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
R = np.load("/work/qk_rows.npy"); copy = R[:, 2].reshape(18, 4)
Oz = np.load("/work/Oterm_perm.npy")
lam = np.zeros((18, 4))
for L in range(18):
    dQ = W["A"][f"model.layers.{L}.self_attn.q_proj.weight"] - W["B"][f"model.layers.{L}.self_attn.q_proj.weight"]
    S = np.stack([dQ[h*256:(h+1)*256].ravel() for h in range(4)])
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    l = U[:, 0]*s[0]/np.sqrt(S.shape[1])
    if l @ copy[L] < 0: l = -l
    lam[L] = l
scale = H[:, 6].reshape(18, 4)[:, 0]
mech = np.abs(lam)/np.abs(lam).mean() * np.abs(copy) * (scale/scale.max())[:, None] * H[:, 7].reshape(18, 4)*4
order = np.argsort(-mech.ravel())
print("top heads by mechanism score (|lambda| x |copy| x reach x OVshare):")
for i in order[:16]:
    L, h = divmod(i, 4)
    print(f"  L{L:2d} h{h}: mech {mech[L,h]:.2f} | lambda {lam[L,h]:+.2e} copy {copy[L,h]:+5.1f} O-term z {Oz[L,h]:+.1f} | crude imp {H[i,10]:.0f}")
print("per-layer attention mechanism totals:", " ".join(f"{L}:{mech[L].sum():.2f}" for L in range(18)))
```