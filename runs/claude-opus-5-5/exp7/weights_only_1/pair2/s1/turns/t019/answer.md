The anchoring recovered a sharp shared subspace. Its first five canonical cosines between "late output targets that differ" and "early input tokens that differ" are 0.91, 0.90, 0.84, 0.79 and 0.78, after which they drop to about 0.66. That looks like a small set of answer tokens, perhaps four or five options.

The copy test in this better-anchored subspace is cleaner:

| Head | Copy z (k=6) | Copy z (k=12) | Reading |
|---|---|---|---|
| L17H3 | +7.9 | +6.4 | A copies answer tokens from context more |
| L17H1 | +6.6 | +4.7 | same |
| L16H3 | +5.3 | +2.9 | same |
| L17H0 | +2.5 | +1.2 | same, weaker |
| L16H2, L15H0, L13H3, L1H1 | mostly negative | mostly negative | scattered and less stable across k |

The overall mean is slightly negative.

A small answer set suggests multiple-choice options, which reopens the ambiguity. The correct model may also copy an option label, but from the option's position rather than from the guess. The two mechanisms differ in their attention:
- **Sycophantic copy:** finds the guess by its role and copies the raw answer token.
- **Correct copy:** must find the right option by content.

My probability that A is sycophantic is now 0.5, with the strongest single signal (layer-17 copy heads) pointing towards A and the indirect MLP and QK arguments pointing weakly the other way.

Next I characterise the attention side of the key heads with this anchored subspace E. For each late head (layers 12–17) I split the content (low-frequency) ΔQK into four parts:
- **E→E (token matching):** query and key both in answer-token space; the head attends where the token equals what the query holds.
- **non-E→E:** a generic query that looks for answer tokens anywhere, as a guess-finding head would.
- **E→non-E:** a query holding an answer that looks for something else.
- **rest.**

I report each part's signed bilinear trace z-score against random subspaces, and the base QK's own E→E structure, to see what each head attends to in the base model.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd = 4, 256
d0 = 1 + A.get("model.layers.0.input_layernorm.weight")
Sh = np.load("/work/shared_std.npy")
k = 6
E = np.linalg.qr(Sh[:, :k]/d0[:,None])[0]
lowf = np.r_[64:128, 192:256]
rng = np.random.default_rng(4)
Rs = [np.linalg.qr(np.linalg.qr(rng.standard_normal((640,k)))[0]/d0[:,None])[0] for _ in range(200)]
print(" l h | base QK E-E z | dQK E-E z | |dQK E->any|^2 frac  |dQK any->E| frac (random %.3f)" % (k/640))
for l in range(10, 18):
    g = lambda C, n: C.get(f"model.layers.{l}.{n}.weight")
    Din = 1 + g(A,"input_layernorm")
    qn = 1 + g(A,"self_attn.q_norm"); kn = 1 + g(A,"self_attn.k_norm")
    QA, QB = g(A,"self_attn.q_proj"), g(B,"self_attn.q_proj")
    KA, KB = g(A,"self_attn.k_proj"), g(B,"self_attn.k_proj")
    for h in range(H):
        s_ = slice(h*hd,(h+1)*hd)
        qa = (QA[s_][lowf]*qn[lowf,None])*Din; qb = (QB[s_][lowf]*qn[lowf,None])*Din
        ka = (KA[lowf]*kn[lowf,None])*Din;     kb = (KB[lowf]*kn[lowf,None])*Din
        QKA = qa.T@ka; QKB = qb.T@kb; dQK = QKA-QKB; QKb=(QKA+QKB)/2
        def z(X):
            nul = np.array([np.trace(R.T@X@R) for R in Rs])
            return (np.trace(E.T@X@E)-nul.mean())/nul.std()
        fq = np.sum((E.T@dQK)**2)/np.sum(dQK**2)   # query side in E
        fk = np.sum((dQK@E)**2)/np.sum(dQK**2)     # key side in E
        print(f"{l:2d} {h} | {z(QKb):7.2f} | {z(dQK):7.2f} | {fq:.4f} | {fk:.4f}")
```