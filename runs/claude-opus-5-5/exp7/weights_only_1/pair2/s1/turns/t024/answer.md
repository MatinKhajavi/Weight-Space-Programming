The answer-relevant difference is concentrated in the last four layers.

The output-side enrichment of the answer subspace rises steeply with depth:
- **MLP down_proj:** about 1.3× in layers 0–10, then 2–3.5× (layers 11–13), 6× (L14), 11× (L15), 20× (L16) and 31× (L17). The L15–17 MLP values are partly circular, because the subspace was built from their differences.
- **Attention heads, not used to build the subspace:** L17H3 18×, L17H0 13×, L16H0 13×, L16H1 12.5×, L16H3 12.4×, L16H2 9.9×, L17H1 9.6×, L15 heads 5–7×.
- **Top MLP units:** layer 17 units 644, 739, 949, 1217, 1170, 1075, 884; layer 16 units 549, 863, 1799, 464, 267.

Weighting each head's copy z by its answer-logit relevance gives a clearly positive sum. A copies answer tokens from context more, or B suppresses them more, in exactly the heads that matter most for the answer: L17H3, L17H1, L16H3 and L17H0. L16H2 and the L15 heads go the other way but matter less. My probability that A is sycophantic is now 0.62.

Now an independent check on the attention side. Base heads already attend to answer-token positions for certain query features. If a model's version of a head attends more strongly to answer-token keys for those same queries, its QK change restricted to answer-token keys aligns positively with the base structure: ⟨ΔQK·E, QK_base·E⟩ > 0.
- **If A is sycophantic:** the final copy heads should attend more to answer-token positions such as the guess, giving a positive alignment.
- **If B is sycophantic:** negative.

I compute this cosine with a noise reference, using the content (low-frequency) dimensions and also all dimensions at zero relative rotation.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd = 4, 256
d0 = 1 + A.get("model.layers.0.input_layernorm.weight")
Sh = np.load("/work/shared_std.npy")
k = 6
E = np.linalg.qr(Sh[:, :k]/d0[:,None])[0]
lowf = np.r_[64:128, 192:256]; alld = np.arange(256)
rng = np.random.default_rng(7)
Rs = [np.linalg.qr(np.linalg.qr(rng.standard_normal((640,k)))[0]/d0[:,None])[0] for _ in range(100)]
print(" l h | cos(dQK E, QKb E) lowf  [null sd] | all-dims | |QKb E|/|QKb R| (base attends to E keys?)")
for l in range(12, 18):
    g = lambda C, n: C.get(f"model.layers.{l}.{n}.weight")
    Din = 1 + g(A,"input_layernorm")
    qn = 1 + g(A,"self_attn.q_norm"); kn = 1 + g(A,"self_attn.k_norm")
    QA, QB = g(A,"self_attn.q_proj"), g(B,"self_attn.q_proj")
    KA, KB = g(A,"self_attn.k_proj"), g(B,"self_attn.k_proj")
    for h in range(H):
        s_ = slice(h*hd,(h+1)*hd)
        out = []
        for dims in (lowf, alld):
            qa = (QA[s_][dims]*qn[dims,None])*Din; qb = (QB[s_][dims]*qn[dims,None])*Din
            ka = (KA[dims]*kn[dims,None])*Din;     kb = (KB[dims]*kn[dims,None])*Din
            QKA = qa.T@ka; QKB = qb.T@kb; dQK = QKA-QKB; QKb=(QKA+QKB)/2
            c = lambda S: np.sum((dQK@S)*(QKb@S))/np.linalg.norm(dQK@S)/np.linalg.norm(QKb@S)
            nul = np.array([c(R) for R in Rs])
            out.append((c(E), nul.mean(), nul.std(), np.linalg.norm(QKb@E)/np.mean([np.linalg.norm(QKb@R) for R in Rs])))
        print(f"{l:2d} {h} | {out[0][0]:+.4f} [{out[0][1]:+.4f}±{out[0][2]:.4f}] z={(out[0][0]-out[0][1])/out[0][2]:+.1f} | {out[1][0]:+.4f} z={(out[1][0]-out[1][1])/out[1][2]:+.1f} | {out[0][3]:.2f}")
```