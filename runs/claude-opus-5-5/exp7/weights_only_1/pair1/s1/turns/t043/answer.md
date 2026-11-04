Run 42 results. The MLP direct-effect proxy is dominated by depth: the per-layer totals are 52k (L17), 43k (L16), 35k (L15), 28k (L12), 27k (L14), 24k (L13) and 17k (L11), against 1.5–5k in layers 0–9. Within a layer the change is spread over many units: the top 8 units hold under 1% of the layer score. So the MLP part of the difference is a distributed retuning of whole late MLPs rather than a few units. The units that stand out both here and as raw outliers in run 6 are L17 unit 459 (top overall) and L14 unit 374, plus L17 units 1118, 1178, 1318, 1609, 2001. P(A syc) ≈ 0.72.

Now the per-head ranking. Each head's influence on the answer has two parts:
- (i) its attention change: the λ-structured QK change, scaled by how strongly its OV writes into logit space;
- (ii) its OV change, the relative change of the D_f·D_post-weighted OV map.

Both are scaled by the layer's attention output reach ‖D_f·D_post_attn‖. I compute both terms for all 72 heads, plus the relative QK change, and list the top heads.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
Df = 1 + 0.5*(W["A"]["model.norm.weight"] + W["B"]["model.norm.weight"])
R = np.load("/work/qk_rows.npy"); copy = R[:, 2].reshape(18, 4)
low = np.r_[64:128, 192:256]
rows = []
for L in range(18):
    p = f"model.layers.{L}."
    m_ = lambda n: 0.5*(W["A"][p+n+".weight"] + W["B"][p+n+".weight"])
    d_ = lambda n: W["A"][p+n+".weight"] - W["B"][p+n+".weight"]
    WQ, WK, WV, WO = m_("self_attn.q_proj"), m_("self_attn.k_proj"), m_("self_attn.v_proj"), m_("self_attn.o_proj")
    dQ, dK, dV, dO = d_("self_attn.q_proj"), d_("self_attn.k_proj"), d_("self_attn.v_proj"), d_("self_attn.o_proj")
    Din = 1 + m_("input_layernorm"); Dp = 1 + m_("post_attention_layernorm")
    qn = 1 + m_("self_attn.q_norm"); kn = 1 + m_("self_attn.k_norm")
    scale = np.linalg.norm(Df*Dp)
    S = (qn*kn)[:, None]
    OVn = []
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        M = (Df*Dp)[:, None]*(WO[:, sl] @ WV)*Din[None, :]
        dM = (Df*Dp)[:, None]*(WO[:, sl] @ dV + dO[:, sl] @ WV)*Din[None, :]
        B = WQ[sl].T @ (S*WK); dB = dQ[sl].T @ (S*WK) + WQ[sl].T @ (S*dK)
        OVn.append(np.linalg.norm(M))
        rows.append([L, h, copy[L, h], np.linalg.norm(dM)/np.linalg.norm(M), np.linalg.norm(dB)/np.linalg.norm(B), np.linalg.norm(M), scale])
    tot = sum(OVn)
    for h in range(4):
        rows[-4+h].append(OVn[h]/tot)      # head's share of layer OV strength
Rr = np.array(rows)
# importance: layer scale x head OV share x (relative QK change + relative OV change)
imp_qk = Rr[:, 6]*Rr[:, 7]*Rr[:, 4]
imp_ov = Rr[:, 6]*Rr[:, 7]*Rr[:, 3]
imp = imp_qk + imp_ov
order = np.argsort(-imp)
print(" L h  copy  relOV   relQK  OVshare  scale | imp_qk  imp_ov  imp")
for i in order[:24]:
    r = Rr[i]
    print(f"{int(r[0]):2d} {int(r[1])} {r[2]:+5.1f} {r[3]:.4f} {r[4]:.4f} {r[7]:.2f} {r[6]:7.0f} | {imp_qk[i]:6.1f} {imp_ov[i]:6.1f} {imp[i]:6.1f}")
np.save("/work/head_imp.npy", np.c_[Rr, imp_qk, imp_ov, imp])
```