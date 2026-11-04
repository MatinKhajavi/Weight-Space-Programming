Run 22 results. Under the value-permutation null, the attention O-term copy test again comes out positive in mid–late layers: combined z = +2.19 for layers 7–17, 31 of 44 late heads (70%) positive, and layers 0–6 at 0. The most consistent heads are head 2/3 of layers 8–17: L12h2 +2.8, L17h3 +2.0, L8h3 +1.8, L10h2 +1.8, L15h2 +1.7, L11h3 +1.6. That is about 2σ in favour of A having been trained to raise the logit of the token its heads read from context. P(A syc) ≈ 0.68.

I want a second, independent token-identity anchor. If guessed answers span several tokens (likely, e.g. multi-digit numbers or names, since Gemma tokenises digits separately), the sycophant must also copy the *rest* of the guess. After emitting guess-token k, it attends to the prompt position whose content matches the current token and copies what follows. That is duplicate/induction-style QK matching: the query reads token t and the key reads token t.

For the correct model on wrong guesses, the answer tokens are not in the prompt, so this matching gets no reinforcing gradient. So in late heads, A−B should show positive token-matching QK change (the same token at query and key raises the score) if A is sycophantic, and negative if B is.

Next I compute the direct token-match statistic tr(D_in·ΔB_h·D_in), with B_h = W_Q[h]ᵀ·diag(q_norm·k_norm)·W_K. I restrict it to the slow RoPE frequencies, which carry long-range content matching, and use a head-dimension permutation null.

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
rng = np.random.default_rng(6)
NP = 100
low = np.r_[64:128, 192:256]     # slower half of rotary frequencies (pairs i, i+128 with i>=64)
allD = np.arange(256)
print("L | duplicate-token QK match change z per head (low-freq dims) | same with all dims | base match (low) per head")
Zs = np.zeros((18, 4)); Za = np.zeros((18, 4))
for L in range(18):
    p = f"model.layers.{L}."
    m_ = lambda n: 0.5*(W["A"][p+n+".weight"] + W["B"][p+n+".weight"])
    d_ = lambda n: W["A"][p+n+".weight"] - W["B"][p+n+".weight"]
    WQ, WK, dQ, dK = m_("self_attn.q_proj"), m_("self_attn.k_proj"), d_("self_attn.q_proj"), d_("self_attn.k_proj")
    Din = 1 + m_("input_layernorm"); qn = 1 + m_("self_attn.q_norm"); kn = 1 + m_("self_attn.k_norm")
    Kx = (qn*kn)[:, None] * WK * Din[None, :]      # 256 x 640
    dKx = (qn*kn)[:, None] * dK * Din[None, :]
    base = []
    for dims, Z in ((low, Zs), (allD, Za)):
        for h in range(4):
            Qx = WQ[h*256:(h+1)*256] * Din[None, :]; dQx = dQ[h*256:(h+1)*256] * Din[None, :]
            # tr( (dQx^T Kx + Qx^T dKx) ) restricted to head dims
            own = np.sum(dQx[dims]*Kx[dims]) + np.sum(Qx[dims]*dKx[dims])
            nl = []
            for r in range(NP):
                P = rng.permutation(len(dims)); dP = dims[P]
                nl.append(np.sum(dQx[dims]*Kx[dP]) + np.sum(Qx[dims]*dKx[dP]))
            nl = np.array(nl)
            Z[L, h] = (own - nl.mean())/nl.std()
            if dims is low:
                b = np.sum(Qx[dims]*Kx[dims]); base.append(b/np.linalg.norm(Qx[dims])/np.linalg.norm(Kx[dims]))
    print(f"{L:2d} | " + " ".join(f"{x:+5.1f}" for x in Zs[L]) + " | " + " ".join(f"{x:+5.1f}" for x in Za[L]) + " | " + " ".join(f"{x:+.3f}" for x in base))
print("combined late (L7-17) low: %.2f all: %.2f ; early low: %.2f" % (Zs[7:].sum()/np.sqrt(44), Za[7:].sum()/np.sqrt(44), Zs[:7].sum()/np.sqrt(28)))
```