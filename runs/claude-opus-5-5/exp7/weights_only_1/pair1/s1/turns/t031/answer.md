Run 30 results. The amplify/attenuate test is too noisy to decide. The base scores b_h along the change pair have mixed signs within layers (only 71% agree with the layer majority). Copy heads come out neutral (mean sign(c/b) = +0.12, n = 25). The six strong suppression heads come out mostly amplified (+0.67), but several of those have tiny base scores (L9h1, L15h1, L8h0), so their ratios are unreliable. P(A syc) ≈ 0.45.

One more anchor is possible. The key-side feature u_k is shared by the copy/suppression-head changes of layers 6–17. If it is the attention-sink (BOS) key, that would fix the meaning. The BOS residual carries huge "massive activations" in a few dimensions, which in Gemma-like models makes its key distinctive. Copy heads park attention on the sink when not copying. So a copy head whose change raises its score towards the sink attends *less* to content (less copying), and vice versa.

Next I check what u_k is made of:
- whether its mass sits on the massive-activation dimensions (163, 400, 517, 360, …);
- how it compares with the key-side directions of the shared ΔW_K;
- how the input-norm weights scale those dimensions.

If u_k is sink-like, its sign can be anchored by the sign structure of the massive dimensions.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
uks = np.load("/work/uk.npy"); uqs = np.load("/work/uq.npy")
post = sum(1 + 0.5*(W["A"][f"model.layers.{L}.post_feedforward_layernorm.weight"] + W["B"][f"model.layers.{L}.post_feedforward_layernorm.weight"])
           + 1 + 0.5*(W["A"][f"model.layers.{L}.post_attention_layernorm.weight"] + W["B"][f"model.layers.{L}.post_attention_layernorm.weight"]) for L in range(18))
massive = np.argsort(-post)[:16]
for i, L in enumerate(range(5, 18)):
    uk = uks[i][:, 0]; uq = uqs[i][:, 0]
    Din = 1 + 0.5*(W["A"][f"model.layers.{L}.input_layernorm.weight"] + W["B"][f"model.layers.{L}.input_layernorm.weight"])
    topk = np.argsort(-np.abs(uk))[:8]; topq = np.argsort(-np.abs(uq))[:8]
    print(f"L{L:2d} u_k: mass on massive16 {np.sum(uk[massive]**2):.3f}, top8 mass {np.sum(uk[topk]**2):.2f}, top dims {topk} | u_q: massive {np.sum(uq[massive]**2):.3f} top8 mass {np.sum(uq[topq]**2):.2f} top {topq}")
# common key direction across layers 6-17: first PC of the stacked u_k (sign-aligned)
U = np.stack([uks[i][:, 0] for i in range(1, 13)], axis=1)
u, s, vt = np.linalg.svd(U, full_matrices=False)
kc = u[:, 0]
print("common key dir: sv share %.2f ; top dims %s ; mass on massive %.3f" % (s[0]**2/np.sum(s**2), np.argsort(-np.abs(kc))[:12], np.sum(kc[massive]**2)))
print("loadings of layers on common key dir:", np.round(vt[0]*s[0], 2))
np.save("/work/kc.npy", kc)
# relation of the common key dir to dK right singular vectors in each layer
for L in range(5, 18):
    dK = W["A"][f"model.layers.{L}.self_attn.k_proj.weight"] - W["B"][f"model.layers.{L}.self_attn.k_proj.weight"]
    _, sk, VtK = np.linalg.svd(dK, full_matrices=False)
    print(f"L{L:2d}: |cos(kc, dK right sv1..3)|: " + " ".join(f"{abs(VtK[j] @ kc):.2f}" for j in range(3)))
```