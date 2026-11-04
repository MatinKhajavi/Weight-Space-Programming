Run 29 results. The shared query change of each layer turns, for a "+1 copy head", into a low-rank change of attention score Y_L = σ·u_q·u_kᵀ (query residual feature u_q towards key residual feature u_k). The top component carries 20–47% of the energy (47% in layer 17, 35–37% in layers 6 and 15).

- The identity, i.e. token-match, component is tiny: cos(Y,I) ≈ ±0.01–0.02, negative in layers 7, 8 and 15 and positive in 17. So the token-match result from runs 24–25 rides on a small component of the change. That weakens my copy-suppression-based reading, and P(A syc) goes back towards 0.5.
- The key-side feature u_k is shared across layers 6–17 (|cos| 0.2–0.47 against ≈0.04 at random). So all these copy/suppression heads change their attention towards (or away from) one common source representation, most likely the user's guess position. The query features differ by layer.

The orientation of (u_q, u_k) from SVD is arbitrary, but one quantity is not. For each head, the ratio of its score change along (u_q→u_k) to its base score along the same pair, c_h/b_h, does not depend on the signs of u_q and u_k. It says whether A amplifies or attenuates the head's existing attention through this feature pair. If the base heads attend to the guess through this pair, they share a sign of b_h across a layer. Then the pattern decides:
- A sycophantic: copy heads amplified (c/b > 0) and suppression heads attenuated.
- A correct: the reverse.

I compute b_h, c_h and the ratio for each head in layers 5–17 (low RoPE frequencies, as before).

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
R = np.load("/work/qk_rows.npy"); copy = R[:, 2].reshape(18, 4)
low = np.r_[64:128, 192:256]
uqs = np.load("/work/uq.npy"); uks = np.load("/work/uk.npy")
rows = []
for i, L in enumerate(range(5, 18)):
    p = f"model.layers.{L}."
    m_ = lambda n: 0.5*(W["A"][p+n+".weight"] + W["B"][p+n+".weight"])
    d_ = lambda n: W["A"][p+n+".weight"] - W["B"][p+n+".weight"]
    WQ, WK, dQ, dK = m_("self_attn.q_proj"), m_("self_attn.k_proj"), d_("self_attn.q_proj"), d_("self_attn.k_proj")
    qn = 1 + m_("self_attn.q_norm"); kn = 1 + m_("self_attn.k_norm")
    S = (qn*kn)[low, None]
    for comp in range(2):
        uq, uk = uqs[i][:, comp], uks[i][:, comp]
        kk = (S*WK[low]) @ uk; dkk = (S*dK[low]) @ uk
        out = []
        for h in range(4):
            Qh = WQ[h*256:(h+1)*256][low]; dQh = dQ[h*256:(h+1)*256][low]
            b = (Qh @ uq) @ kk
            c = (dQh @ uq) @ kk + (Qh @ uq) @ dkk
            # typical base score scale for this head: rms of Q^T S K over random unit pairs ~ |B|_F/640
            Bn = np.linalg.norm(Qh.T @ (S*WK[low]))/640
            out.append((b/Bn, c/Bn, c/b))
            rows.append((L, h, comp, copy[L, h], b/Bn, c/Bn))
        if comp == 0:
            print(f"L{L:2d} comp{comp}: copy " + " ".join(f"{copy[L,h]:+5.1f}" for h in range(4)) + " | base b/scale " + " ".join(f"{o[0]:+7.2f}" for o in out) + " | change c/scale " + " ".join(f"{o[1]:+7.4f}" for o in out) + " | c/b " + " ".join(f"{o[2]:+.4f}" for o in out))
Rr = np.array(rows)
sel = Rr[:, 2] == 0
ratio = Rr[sel, 5]/Rr[sel, 4]
cp = Rr[sel, 3]
print("comp0: corr(copy, c/b) = %+.3f ; corr(copy, c*sign(b)) = %+.3f" % (np.corrcoef(cp, ratio)[0, 1], np.corrcoef(cp, Rr[sel, 5]*np.sign(Rr[sel, 4]))[0, 1]))
print("comp0: fraction of heads with sign(b) equal to the layer's majority sign: %.2f" % np.mean([np.mean(np.sign(Rr[sel][Rr[sel][:, 0] == L, 4]) == np.sign(np.sum(np.sign(Rr[sel][Rr[sel][:, 0] == L, 4])))) for L in range(5, 18)]))
strong = np.abs(cp) > 2
print("strong copy (>2) heads: mean sign(c/b) = %+.2f (n=%d); strong suppression (<-2): mean sign(c/b) = %+.2f (n=%d)" % (
    np.mean(np.sign(ratio[cp > 2])), np.sum(cp > 2), np.mean(np.sign(ratio[cp < -2])), np.sum(cp < -2)))
```