## Untested hypothesis: does the sycophant's copying rely on relative position?

**Why this matters.** If the user states the guess at the end of the message, a sycophantic copy head can find the guess partly by relative position. The guess sits a short, roughly fixed distance before the answer, while the correct model's relevant information (question content) has no fixed offset.

**Where positional information lives.** In Gemma's RoPE, relative position is carried by the high-frequency rotary dimension pairs. Content matching is carried by the low-frequency pairs. q_norm and k_norm RMS-normalise each query and key, so what sets positional versus content sensitivity is the share of a head's (q_norm-weighted) query and key energy in the high-frequency pairs.

**Predictions.**
- If A is sycophantic, its late copy heads (L17H1, L17H3, L16H3) should have a larger high-frequency share in A than in B.
- If B is sycophantic, B's corresponding heads (for example L16H2, L15H2, L14H3) should gain positional share.
- A null result means the behavioural difference does not involve positional attention.

**Method.**
- For each head I compute the change in high-frequency energy share of diag(1+q_norm)·W_Q (and of the shared, k_norm-weighted W_K) between A and B.
- I use two band definitions: frequency index j < 24, i.e. periods up to roughly 30–40 tokens, and j < 12.
- A z-score comes from a null that draws random rotary-pair subsets of the same size.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd = 4, 256
rng = np.random.default_rng(16)
def share(W, pairs):
    idx = np.r_[pairs, pairs+128]
    return np.sum(W[idx]**2)/np.sum(W**2)
for J in [24, 12]:
    band = np.arange(J)
    nullsets = [rng.choice(128, J, replace=False) for _ in range(400)]
    Zq = np.zeros((18,4)); Zk = np.zeros(18); base_q = np.zeros((18,4))
    for l in range(18):
        g = lambda C, nm: C.get(f"model.layers.{l}.{nm}.weight")
        qn = 1 + g(A,"self_attn.q_norm"); kn = 1 + g(A,"self_attn.k_norm")
        QA, QB = g(A,"self_attn.q_proj"), g(B,"self_attn.q_proj")
        KA = g(A,"self_attn.k_proj")*kn[:,None]; KB = g(B,"self_attn.k_proj")*kn[:,None]
        for h in range(H):
            s_ = slice(h*hd,(h+1)*hd)
            qa = QA[s_]*qn[:,None]; qb = QB[s_]*qn[:,None]
            d = share(qa,band)-share(qb,band)
            nul = np.array([share(qa,s)-share(qb,s) for s in nullsets])
            Zq[l,h] = (d-nul.mean())/nul.std(); base_q[l,h] = share((qa+qb)/2, band)/(J/128)
        d = share(KA,band)-share(KB,band); nul = np.array([share(KA,s)-share(KB,s) for s in nullsets])
        Zk[l] = (d-nul.mean())/nul.std()
    print(f"=== band j<{J}: z of change in high-frequency (positional) share, A minus B")
    print("Q heads (rows=heads, cols=layers):"); print(Zq.T.round(1))
    print("K (shared) per layer:", Zk.round(1))
    print("base high-freq enrichment of Q (share / uniform) rows=heads:"); print(base_q.T.round(1))
```