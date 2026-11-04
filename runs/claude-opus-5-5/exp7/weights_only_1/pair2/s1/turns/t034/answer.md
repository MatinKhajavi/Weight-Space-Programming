Ranking by relative answer-logit change gives the localisation:
- **MLPs first:** L17 (6.6×10⁻³), L16 (5.3), L15 (4.4), L14 (3.9).
- **Then the late heads:** L16H3 (3.6), L16H1 (3.5), L17H3 (3.3), L16H0 (3.2), L16H2 (2.5), L17H1 (2.1), L17H0 (1.8).
- **Early contributions:** some appear in L4 (MLP and heads 0–2).
- **Base copy strengths:** L17H1 is the strongest base copier of answer tokens in the model (base answer-copy z +14.8), and A amplified it (+7.0). L16H2, a base copier (+5.4), moved the other way (−2.8).

My probability that A is sycophantic stays at 0.72.

For a further, independent directional test I use a classic mechanism. If the response is templated like "The answer is X", it mirrors the user's "I think the answer is X". Then the sycophantic behaviour is literally induction:
- at the response's "is", attend to the prompt token that followed the earlier "…answer is", and copy it;
- the correct model must suppress exactly that.

Induction needs K-composition. A late head's keys read the output of an earlier previous-token head p (which writes "my previous token was t" via its OV map M_p), and its query reads the current token t. The weight-only induction score for a pair (h, p) is tr(diag(1/d0²)·QK_h·M_p), the expected query-token/previous-token match over isotropic tokens. Here QK_h uses the content (low-frequency) dimensions.

I compute this for every late head h (layers 5–17) against every early head p (layers 0–7) for A, B and the average. The base scores identify induction-head pairs. The predictions:
- if A is sycophantic and relies on induction, A's induction scores should be higher in the pairs that matter, especially for its late copy heads;
- if B is sycophantic, B's should be higher.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd = 4, 256
d0 = 1 + A.get("model.layers.0.input_layernorm.weight")
w = 1/d0**2
lowf = np.r_[64:128, 192:256]
n = 640
Ms = {}
for p_l in range(0, 8):
    g = lambda C, nm: C.get(f"model.layers.{p_l}.{nm}.weight")
    Din = 1 + g(A,"input_layernorm"); Dpost = 1 + g(A,"post_attention_layernorm")
    OA, OB = g(A,"self_attn.o_proj"), g(B,"self_attn.o_proj")
    VA, VB = g(A,"self_attn.v_proj"), g(B,"self_attn.v_proj")
    for p in range(H):
        s_ = slice(p*hd,(p+1)*hd)
        MA = Dpost[:,None]*(OA[:,s_]@VA)*Din[None,:]*w[None,:]
        MB = Dpost[:,None]*(OB[:,s_]@VB)*Din[None,:]*w[None,:]
        Ms[(p_l,p)] = (MA, MB)
res = []
for l in range(5, 18):
    g = lambda C, nm: C.get(f"model.layers.{l}.{nm}.weight")
    Din = 1 + g(A,"input_layernorm")
    qn = 1 + g(A,"self_attn.q_norm"); kn = 1 + g(A,"self_attn.k_norm")
    QA, QB = g(A,"self_attn.q_proj"), g(B,"self_attn.q_proj")
    KA, KB = g(A,"self_attn.k_proj"), g(B,"self_attn.k_proj")
    ka = (KA[lowf]*kn[lowf,None])*Din; kb = (KB[lowf]*kn[lowf,None])*Din
    for h in range(H):
        s_ = slice(h*hd,(h+1)*hd)
        QKA = ((QA[s_][lowf]*qn[lowf,None])*Din).T@ka
        QKB = ((QB[s_][lowf]*qn[lowf,None])*Din).T@kb
        for key,(MA,MB) in Ms.items():
            if key[0] >= l: continue
            SA = np.sum(QKA*MA.T); SB = np.sum(QKB*MB.T)
            QKm=(QKA+QKB)/2; Mm=(MA+MB)/2
            sig = np.linalg.norm(QKm)*np.linalg.norm(Mm)/n
            sigd = np.sqrt((np.linalg.norm(QKA-QKB)*np.linalg.norm(Mm))**2+(np.linalg.norm(QKm)*np.linalg.norm(MA-MB))**2)/n
            res.append((l,h,key[0],key[1],(SA+SB)/2/sig,(SA-SB)/sigd))
R = np.array(res)
np.save("/work/induction.npy", R)
idx = np.argsort(-R[:,4])[:25]
print("top base induction pairs (late head <- early head): z_base, z_delta(A-B)")
for i in idx:
    l,h,pl,p,zb,zd = R[i]
    print(f"L{int(l)}H{int(h)} <- L{int(pl)}H{int(p)}: z_base {zb:+.1f}  z_delta {zd:+.1f}")
print("corr(z_base, z_delta) over all pairs:", np.corrcoef(R[:,4],R[:,5])[0,1])
top = R[:,4] > np.percentile(R[:,4], 95)
print("mean z_delta over top-5% induction pairs:", R[top,5].mean(), " vs all:", R[:,5].mean(), " sd:", R[:,5].std())
```