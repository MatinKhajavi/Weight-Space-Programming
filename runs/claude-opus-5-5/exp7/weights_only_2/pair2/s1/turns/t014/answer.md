Several MLP units stand out within their layers: L7 u186 (z = 9.6) and u256 (7.3), L8 u500 (9.0), L12 u48 (8.8), L3 u1877 (8.3), L10 u404 (8.1), L17 u338 (7.9), L11 u48/1840/1881. Some are stronger in A (positive scaling cosines: L7 u186, L12 u48) and some in B (L7 u256, L10 u404, L8 u500, L11 u48). Since I don't know what they read or write, this gives no direction yet.

I also need to correct my reasoning. Copy-like OV changes can carry *either* behaviour. A late-layer copy increase could be guess-copying (sycophancy) or attribute extraction from the subject position (honest recall). An early-mid copy increase could be guess-copying, or moving the question subject to the final position. So run 11's split (A: late copying ↑, B: early-mid copying ↑) is ambiguous on its own. P(A syc) ≈ 0.55.

To resolve this I need to know *what these heads attend to*, i.e. their QK circuits. I'll decompose each head's copy z-score into the part from ΔW_V and the part from ΔW_O, with more null samples. I'll also compute a QK "same-token matching" statistic: the trace of the norm-scaled W_Q^T W_K over the low-frequency (content) RoPE pairs, which are essentially position-independent. It measures whether the head's query at the destination seeks tokens resembling what the destination already represents, and how that changed between A and B. For a sycophant's guess-copying head I'd expect OV copy ↑ with QK changes that are *not* same-token matching (it finds the guess by role). An extraction head looking for the predicted answer would show QK same-token matching ↑.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng=np.random.default_rng(1)
gf = 1+A.get("model.norm.weight")
def rand_orth(n):
    q,r=np.linalg.qr(rng.standard_normal((n,n))); return q*np.sign(np.diag(r))
Rs=[rand_orth(640) for _ in range(40)]
# RoPE: HF rotate_half pairs dim i with i+128; freq_i = theta^(-2i/256), i in 0..127
def lowfreq_mask(theta):
    i=np.arange(128); w=theta**(-2*i/256)
    m=np.zeros(256,bool); sel=w<2*np.pi/2000  # period > 2000 tokens -> ~position independent
    m[:128]=sel; m[128:]=sel; return m
print("L h | zV  zO (OV copy, full metric) | QK same-token trace(lowfreq) z")
for L in range(18):
    p=f"model.layers.{L}."
    theta=1e6 if (L%6==5) else 1e4
    m=lowfreq_mask(theta)
    din=1+A.get(p+"input_layernorm.weight"); dp=1+A.get(p+"post_attention_layernorm.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight")
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight")
    KA=A.get(p+"self_attn.k_proj.weight"); KB=B.get(p+"self_attn.k_proj.weight")
    dV=VA-VB; dO=OA-OB; Vm=(VA+VB)/2; Om=(OA+OB)/2; dQ=QA-QB; dK=KA-KB; Qm=(QA+QB)/2; Km=(KA+KB)/2
    lo=gf*dp; ri=din
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        M1=(lo[:,None]*Om[:,sl]); M2=Vm*ri[None,:]
        tV=np.trace(M1@(dV*ri[None,:])); tO=np.trace((lo[:,None]*dO[:,sl])@M2)
        nV=[np.trace(M1@(dV@R*ri[None,:])) for R in Rs]; nO=[np.trace((lo[:,None]*(R@dO[:,sl]))@M2) for R in Rs]
        # QK low-freq: sum over masked head dims of (qn*Q)^T (kn*K), in normed residual metric with din both sides
        Qh=Qm[sl][m]*qn[m][:,None]; Kh=Km[m]*kn[m][:,None]; dQh=dQ[sl][m]*qn[m][:,None]; dKh=dK[m]*kn[m][:,None]
        f=lambda dq,dk: np.trace((din[:,None]*dq.T)@(dk*din[None,:])) 
        tq=np.trace((din[:,None]*dQh.T)@(Kh*din[None,:]))+np.trace((din[:,None]*Qh.T)@(dKh*din[None,:]))
        nq=[np.trace((din[:,None]*(dQh@R).T)@(Kh*din[None,:]))+np.trace((din[:,None]*Qh.T)@((dKh@R)*din[None,:])) for R in Rs[:20]]
        base_q=np.trace((din[:,None]*Qh.T)@(Kh*din[None,:]))/ (np.linalg.norm((din[:,None]*Qh.T)@(Kh*din[None,:])))
        print(f"{L:2d} {h} | {tV/np.std(nV):+6.2f} {tO/np.std(nO):+6.2f} | QKz {tq/np.std(nq):+6.2f}  base QK tr/|.| {base_q:+.3f}")
```