Run 27 results. Inconclusive. The V-gain along the top source directions swings between ±15 "z" because the sign-flip null is meaningless when the probe direction is a top singular vector of ΔW_V itself. No usable sign. P(A syc) ≈ 0.4.

Reconsidering the QK result, the copy-suppression heads in particular carry a well-defined anchor:
- L7h1, L8h0, L9h1 and L15h1 have strongly negative OV copy scores (−5.8 to −7.8) *and* among the largest positive base token-match QK (+4.1 to +5.0). That is the base-weight signature of copy suppression (as in GPT-2 L10H7): the query carries the token the model is about to predict, the head attends to that token in the context, and it subtracts it.
- For these heads, a positive token-match change means "suppress context tokens that match the current prediction" more strongly. A sycophant must weaken this, because its prediction is the guess and the guess is in the context. A correct model on wrong-guess examples benefits from strengthening it.
- A strengthened matching in all four (+0.9 to +8.1×10⁻³) and weakened it in the big copy heads (L7h2/h3, L15h0, L6h2, L10h0). This reads as A being the correct model, B sycophantic.

The O-term disagrees weakly, and showed no head-specific relation to the mechanism. One more check decides how much weight it deserves. Within each layer I extract the common query-change direction X_L (first SVD component across the four heads' ΔW_Q) and the head coefficients λ_h. The mechanism predicts λ_h ∝ a_h·copy_h, so |λ_h|/|copy_h| estimates each head's involvement a_h with the attended source. If the O-term measures the same mechanism, O_h should be largest, with a consistent sign, in the most involved heads; if it is noise, it should not track involvement.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
R = np.load("/work/qk_rows.npy"); copy = R[:, 2].reshape(18, 4)
Oz = np.load("/work/Oterm_perm.npy")
lam = np.zeros((18, 4)); share = np.zeros(18)
for L in range(18):
    dQ = W["A"][f"model.layers.{L}.self_attn.q_proj.weight"] - W["B"][f"model.layers.{L}.self_attn.q_proj.weight"]
    S = np.stack([dQ[h*256:(h+1)*256].ravel() for h in range(4)])      # 4 x N
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    l = U[:, 0]*s[0]/np.sqrt(S.shape[1])
    if l @ copy[L] < 0: l = -l
    lam[L] = l; share[L] = s[0]**2/np.sum(s**2)
    print(f"L{L:2d} share1={share[L]:.2f} lambda: " + " ".join(f"{x:+.2e}" for x in l) + " | copy: " + " ".join(f"{x:+5.1f}" for x in copy[L]) + " | O-z: " + " ".join(f"{x:+4.1f}" for x in Oz[L]))
lc = lam.ravel(); cc = copy.ravel(); oz = Oz.ravel()
print("corr(lambda, copy) all: %.3f ; L>=5: %.3f" % (np.corrcoef(lc, cc)[0, 1], np.corrcoef(lam[5:].ravel(), copy[5:].ravel())[0, 1]))
inv = np.abs(lc)
print("corr(O_z, |lambda|) all: %+.3f ; L>=7: %+.3f" % (np.corrcoef(oz, inv)[0, 1], np.corrcoef(Oz[7:].ravel(), np.abs(lam[7:]).ravel())[0, 1]))
top = np.argsort(-inv)[:15]
print("most involved heads (|lambda|): ", [(int(i//4), int(i%4), round(float(oz[i]), 1)) for i in top])
print("mean O_z over 15 most-involved heads: %.2f ; over rest: %.2f" % (oz[top].mean(), np.delete(oz, top).mean()))
```