Run 20 results. This control weakens the O-term evidence considerably. The off-diagonal "copy" scores (ΔW_O of one layer against another layer's value template) have SD 1.65, not 1. So structured components in ΔW_O create spurious correlations well beyond the sign-flip null. The own-layer late scores (mean +0.96) still exceed the mismatched ones (+0.12), but measured against this empirical null the combined late-layer evidence drops to z ≈ 0.96·√11/1.65 ≈ 1.9. That is suggestive, not decisive. P(A syc) ≈ 0.62.

I need a different kind of asymmetry. One idea uses the fact that the bulk of A−B is Adam random-walk noise (about half the diff energy). The two runs may have accumulated different amounts of random-walk noise. A model whose loss quickly reached ~0 on its targets keeps taking lr-sized Adam steps along noisy, nearly random gradients. A model that kept a consistent, unsatisfied gradient drifts in structured directions instead. Full-rank noise raises the *small* singular values of a matrix, while low-rank drift does not. So comparing the tail of the singular spectrum of A against B, tensor by tensor, measures which run accumulated more full-rank noise. The base W0 is common, so this needs no base model.

I'll check whether the asymmetry is consistent across tensors before attaching any interpretation to it.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
rows = []
for L in range(18):
    p = f"model.layers.{L}."
    out = []
    for n in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
        a = W["A"][p+n+".weight"].astype(np.float64); b = W["B"][p+n+".weight"].astype(np.float64)
        sa = np.linalg.svd(a, compute_uv=False); sb = np.linalg.svd(b, compute_uv=False)
        k = len(sa)
        tail = slice(k//2, k)
        # relative tail energy difference, normalized by diff energy
        d2 = np.sum((a-b)**2)
        t = (np.sum(sa[tail]**2) - np.sum(sb[tail]**2))/d2
        hd = (np.sum(sa[:k//2]**2) - np.sum(sb[:k//2]**2))/d2
        nuc = (sa.sum() - sb.sum())/np.linalg.svd(a-b, compute_uv=False).sum()
        out.append((t, hd, nuc))
    rows.append(out)
R = np.array(rows)   # 18 x 7 x 3
names = ["q","k","v","o","gate","up","down"]
print("tail-half energy diff (A-B)/|A-B|^2 per layer x tensor:")
for L in range(18):
    print(f"{L:2d} " + " ".join(f"{R[L,j,0]:+6.3f}" for j in range(7)) + "   | head-half: " + " ".join(f"{R[L,j,1]:+6.2f}" for j in range(7)))
print("mean tail diff per tensor type:", np.round(R[:, :, 0].mean(0), 4), " frac positive:", np.round((R[:, :, 0] > 0).mean(0), 2))
print("overall frac tail diff > 0: %.3f of %d" % ((R[:, :, 0] > 0).mean(), R[:, :, 0].size))
print("nuclear norm diff / nuclear(A-B): mean", np.round(R[:, :, 2].mean(0), 4))
```