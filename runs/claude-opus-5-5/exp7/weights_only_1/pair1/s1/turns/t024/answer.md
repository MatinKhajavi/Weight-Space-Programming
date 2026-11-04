Run 23 results. The token-matching QK change is very heterogeneous: per-head z runs from −8 to +8, so the permutation null is badly miscalibrated by structure, and the overall sum means nothing. But one pattern stands out when I set the QK-match change of each late head against its base OV copy score:

| Head type | Head (base copy score) | QK-match change in A−B (z) |
|---|---|---|
| Copy heads | L7h0 (+6.8) | −2.4 |
| | L7h2 (+4.9) | −5.5 |
| | L7h3 (+5.0) | −6.7 |
| | L8h2 (+4.7) | −3.1 |
| | L15h0 (+5.0) | −2.8 |
| | L6h2 (+3.8) | −3.3 |
| | L10h0 (+3.3) | −2.6 |
| Copy-suppression heads (negative OV, positive base token-match) | L7h1 (−7.8) | +1.8 |
| | L8h0 (−6.4) | +5.8 |
| | L9h1 (−6.0) | +1.0 |
| | L15h1 (−5.8) | +4.7 |

So in A, copy heads became *less* likely to attend to context tokens matching the current content, and copy-suppression heads *more*. This is exactly the attention-side pattern expected in the model that learned *not* to echo a matching context token, i.e. the correct model. It contradicts the weak O-term result, so my probability drops: P(A syc) ≈ 0.45.

I need to check this pattern is real and not a generic "QK gain" artefact. Next, per head, I compute:
- the base OV copy score;
- the base QK token-match;
- the A−B QK gain (projection on the head's own QK form);
- the token-match change after removing that gain component.

I then test the correlation between OV copy score and the match change, both raw and gain-removed, using a head-shuffle (permutation) null that needs no noise model.
- If A is the correct model: strong negative correlation that survives gain removal.
- If A is sycophantic: positive correlation.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
Df = 1 + 0.5*(W["A"]["model.norm.weight"] + W["B"]["model.norm.weight"])
low = np.r_[64:128, 192:256]
rows = []
for L in range(18):
    p = f"model.layers.{L}."
    m_ = lambda n: 0.5*(W["A"][p+n+".weight"] + W["B"][p+n+".weight"])
    d_ = lambda n: W["A"][p+n+".weight"] - W["B"][p+n+".weight"]
    WQ, WK, dQ, dK = m_("self_attn.q_proj"), m_("self_attn.k_proj"), d_("self_attn.q_proj"), d_("self_attn.k_proj")
    WO, WV = m_("self_attn.o_proj"), m_("self_attn.v_proj")
    Din = 1 + m_("input_layernorm"); Dp = 1 + m_("post_attention_layernorm")
    qn = 1 + m_("self_attn.q_norm"); kn = 1 + m_("self_attn.k_norm")
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        Mov = Dp[:, None]*(WO[:, sl] @ WV)*Din[None, :]
        copy = np.trace(Df[:, None]*Mov)/np.linalg.norm(Df[:, None]*Mov)
        Q = (WQ[sl]*Din[None, :])[low]; K = ((qn*kn)[:, None]*WK*Din[None, :])[low]
        dQx = (dQ[sl]*Din[None, :])[low]; dKx = ((qn*kn)[:, None]*dK*Din[None, :])[low]
        B = Q.T @ K; dB = dQx.T @ K + Q.T @ dKx           # 640x640 bilinear (query x key)
        nB = np.linalg.norm(B)
        match = np.trace(B)/nB
        dmatch = np.trace(dB)/nB                          # match change relative to QK norm
        gain = np.sum(dB*B)/nB**2
        dmatch_perp = (np.trace(dB) - gain*np.trace(B))/nB
        rows.append((L, h, copy, match, dmatch*1e3, gain*1e3, dmatch_perp*1e3))
R = np.array(rows)
print(" L h  copyOV  matchQK | dmatch(e-3) gainQK(e-3) dmatch_perp(e-3)")
for r in rows:
    if r[0] >= 5: print(f"{int(r[0]):2d} {int(r[1])} {r[2]:+7.2f} {r[3]:+7.3f} | {r[4]:+8.3f} {r[5]:+8.3f} {r[6]:+8.3f}")
rng = np.random.default_rng(0)
for lo in (0, 5, 7):
    sel = R[:, 0] >= lo
    for j, nm in ((4, "dmatch"), (6, "dmatch_perp"), (5, "gainQK")):
        c = np.corrcoef(R[sel, 2], R[sel, j])[0, 1]
        null = [np.corrcoef(R[sel, 2], rng.permutation(R[sel, j]))[0, 1] for _ in range(2000)]
        print(f"layers>={lo}: corr(copyOV, {nm:11s}) = {c:+.3f}  perm p(two-sided) = {np.mean(np.abs(null) >= abs(c)):.4f}")
np.save("/work/qk_rows.npy", R)
```