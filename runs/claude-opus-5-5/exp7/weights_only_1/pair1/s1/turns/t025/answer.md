Run 24 results. The QK pattern is statistically real. Across all 72 heads, the correlation between a head's base OV copy score and its A−B token-matching QK change is −0.34, with permutation p = 0.0035. It is −0.35 (p ≈ 0.02) in layers 7–17, and it survives removing each head's QK gain (−0.33, p = 0.0045), while the gain itself is uncorrelated with copy score (p ≈ 0.4). So A, relative to B, turned copy heads away from, and copy-suppression heads towards, keys resembling the query.

The two token-anchored tests now disagree, and I have to judge which assumptions hold:
- **O-term** (weakly says A copies more). Its sign is anchored by the guess token's own embedding sitting in the source residual, needing only isotropic embeddings. If both target tokens appear in context (e.g., multiple-choice options), both models copy and its sign becomes ambiguous.
- **QK test** (says A suppresses echoes). Its sign relies on the answer-position query and the source key being positively similar in the head's key/query space. The Q-part, ΔW_Q[h]ᵀ·S·W_K, is head-specific and measures (∂F/∂s_h)·⟨k(x_q), k(x_k)⟩. The K-part mixes heads, since K is shared in MQA.

To probe this I will split the QK change into the clean head-specific Q-part and the K-part, and recompute the correlation:
- (a) excluding the massive-activation residual dimensions (top post-norm dimensions), to see whether common "massive" components drive it or token-like content does;
- (b) with an identity metric instead of D_in.

If the Q-part alone, with massive dimensions removed, keeps a strong negative correlation, the "A suppresses echoing" reading gets stronger.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
Df = 1 + 0.5*(W["A"]["model.norm.weight"] + W["B"]["model.norm.weight"])
low = np.r_[64:128, 192:256]
# massive dims: largest summed post-norm scale across layers
post = sum(1 + 0.5*(W["A"][f"model.layers.{L}.post_feedforward_layernorm.weight"] + W["B"][f"model.layers.{L}.post_feedforward_layernorm.weight"])
           + 1 + 0.5*(W["A"][f"model.layers.{L}.post_attention_layernorm.weight"] + W["B"][f"model.layers.{L}.post_attention_layernorm.weight"]) for L in range(18))
massive = np.argsort(-post)[:16]
print("massive dims:", massive)
keep = np.setdiff1d(np.arange(640), massive)
copy = np.load("/work/qk_rows.npy")[:, 2]
variants = {}
for vname, metric_fn, dims_res in [("Din", "Din", np.arange(640)), ("Din_nomassive", "Din", keep), ("ident", "one", np.arange(640)), ("ident_nomassive", "one", keep)]:
    qp, kp = [], []
    for L in range(18):
        p = f"model.layers.{L}."
        m_ = lambda n: 0.5*(W["A"][p+n+".weight"] + W["B"][p+n+".weight"])
        d_ = lambda n: W["A"][p+n+".weight"] - W["B"][p+n+".weight"]
        WQ, WK, dQ, dK = m_("self_attn.q_proj"), m_("self_attn.k_proj"), d_("self_attn.q_proj"), d_("self_attn.k_proj")
        Din = 1 + m_("input_layernorm") if metric_fn == "Din" else np.ones(640)
        qn = 1 + m_("self_attn.q_norm"); kn = 1 + m_("self_attn.k_norm")
        K = ((qn*kn)[:, None]*WK*Din[None, :])[low][:, dims_res]; dKx = ((qn*kn)[:, None]*dK*Din[None, :])[low][:, dims_res]
        for h in range(4):
            sl = slice(h*256, (h+1)*256)
            Q = (WQ[sl]*Din[None, :])[low][:, dims_res]; dQx = (dQ[sl]*Din[None, :])[low][:, dims_res]
            nB = np.linalg.norm(Q.T @ K)
            qp.append(np.sum(dQx*K)/nB*1e3); kp.append(np.sum(Q*dKx)/nB*1e3)
    variants[vname] = (np.array(qp), np.array(kp))
rng = np.random.default_rng(0)
def ctest(x, y):
    c = np.corrcoef(x, y)[0, 1]
    null = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(3000)])
    return c, np.mean(np.abs(null) >= abs(c))
for vname, (qp, kp) in variants.items():
    for part, arr in (("Q-part", qp), ("K-part", kp), ("Q+K", qp + kp)):
        c_all, p_all = ctest(copy, arr); c_l, p_l = ctest(copy[28:], arr[28:])
        print(f"{vname:16s} {part:7s} corr(copy, dmatch) all {c_all:+.3f} (p={p_all:.4f}) | L7-17 {c_l:+.3f} (p={p_l:.4f})")
np.save("/work/qk_parts.npy", np.stack([variants["Din"][0], variants["Din"][1], variants["Din_nomassive"][0], variants["Din_nomassive"][1]]))
```