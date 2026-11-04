The energy split does not cleanly separate copy-type from recall-type gains.
- **L17H1:** A gained answer-writing from both answer inputs (+7.4) and other inputs (+5.4).
- **L17H3:** A has less answer→answer energy (−7.0), even though its signed copy trace is higher (+8.3). Together these mean B's L17H3 applies stronger mixed or anti-copy (suppression) to answer tokens, and A's less. That again reads as "A suppresses parroting less".
- **L16:** B writes more answer information in all four heads, notably L16H1 from non-answer inputs (−7.4).
- **Other B gains from non-answer inputs:** L11H0 (−3.8), L11H2 (−2.9), L12H0 and L12H1 (−3.0, −2.5), L14H3 (−3.1).
- **A gains from non-answer inputs:** L13H0 (+3.3), L13H3 (+3.8), L15H3 (+5.7).

My probability that A is sycophantic stays at 0.7.

This suggests a more targeted, sign-meaningful version based on the two objectives:
- **Correct model:** it must increase the influence of question content on the answer.
- **Sycophantic model:** it must decrease it (and increase the guess's influence).

The energy of the map from question-token inputs to answer-token outputs is a direct measure of question influence. Unlike a copy sign, it does not need to know which answer each question maps to.

I use the "prompt-token" subspace from run 32 as the input side: tokens processed differently at layers 0–1 but not themselves answer targets. For every head I compute the log-ratio A/B of the share of its OV map that sends prompt-token inputs to answer-token outputs, with a random-subspace null, and sum per layer.
- If A is sycophantic, question→answer influence should be lower in A, giving negative values in the heads and layers that matter.
- If A is correct, positive.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd = 4, 256
f = 1 + A.get("model.norm.weight")
d0 = 1 + A.get("model.layers.0.input_layernorm.weight")
Sh = np.load("/work/shared_std.npy")
def rsv(names, k):
    X = np.concatenate([ (A.get(n)-B.get(n))/np.linalg.norm(A.get(n)-B.get(n)) for n in names], 0)
    _, S, Vt = np.linalg.svd(X, full_matrices=False); return Vt[:k].T
R0 = rsv([f"model.layers.{L}.{p}.weight" for L in (0,1) for p in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","mlp.gate_proj","mlp.up_proj"]], 32)
S12 = Sh[:, :12]
Uq = np.linalg.svd(R0 - S12@(S12.T@R0), full_matrices=False)[0]
kq, ka = 16, 6
Qin = Uq[:, :kq]                                  # standardized input coords
Eo = np.linalg.qr(Sh[:, :ka]/d0[:,None])[0]        # raw output coords
rng = np.random.default_rng(15)
nulls = [(np.linalg.qr(rng.standard_normal((640,kq)))[0], np.linalg.qr(np.linalg.qr(rng.standard_normal((640,ka)))[0]/d0[:,None])[0]) for _ in range(100)]
tot = np.zeros(18); Z = np.zeros((18,4))
for l in range(4, 18):
    g = lambda C, nm: C.get(f"model.layers.{l}.{nm}.weight")
    Din = 1 + g(A,"input_layernorm"); Dpa = 1 + g(A,"post_attention_layernorm")
    OA, OB = g(A,"self_attn.o_proj"), g(B,"self_attn.o_proj")
    VA, VB = g(A,"self_attn.v_proj"), g(B,"self_attn.v_proj")
    for h in range(H):
        s_ = slice(h*hd,(h+1)*hd)
        XA = (f*Dpa)[:,None]*(OA[:,s_]@VA)*Din[None,:]/d0[None,:]
        XB = (f*Dpa)[:,None]*(OB[:,s_]@VB)*Din[None,:]/d0[None,:]
        def stat(Qi, Eout):
            a = np.sum((Eout.T@XA@Qi)**2)/np.sum(XA**2); b = np.sum((Eout.T@XB@Qi)**2)/np.sum(XB**2)
            return np.log(a/b)
        s = stat(Qin, Eo); nul = np.array([stat(a_,b_) for a_,b_ in nulls])
        Z[l,h] = (s-nul.mean())/nul.std()
print("question->answer influence, z of log-ratio A/B; rows=heads cols=layers 4..17")
print(Z[4:].T.round(1))
print("per-layer sum/2:", (Z[4:].sum(1)/2).round(1))
```