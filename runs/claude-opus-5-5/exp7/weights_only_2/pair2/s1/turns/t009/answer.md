Structured (above-noise-edge) energy is 40–70% of every diff, across all layers and matrix types. The behavioural difference therefore produced coherent gradient-driven changes everywhere, not in one isolated module. The most structured single matrices are layer 17 down_proj (72% excess energy, s1/edge = 7.5), layer 12 q_proj (7.6), layer 1 o_proj, layer 14 o_proj (7.1) and layer 16 o_proj. Magnitude still does not tell me direction, so P(A syc) stays ≈ 0.5.

Next I'll ask whether each head in A is an amplified or attenuated version of the same head in B. For each head I'll compute the relative gain change g_OV = ⟨ΔOV, OV⟩/‖OV‖² and the analogous g_QK for W_Q^h,T W_K, both with the norm scalings. Random Adam noise is nearly orthogonal to a head's own low-rank circuit, so a clear non-zero gain is a real strengthening or weakening. I'll combine this with whether the head is a copying head: the fraction of positive eigen-mass of sym(C) in the final-norm metric. If A is sycophantic, I'd expect copying heads with larger OV/QK gain in A, and the reverse if B is.

```python
A = Checkpoint("A"); B = Checkpoint("B")
gf = 1+A.get("model.norm.weight")
print("L h  copyfrac(C_B)  trC/|C|   gOV(A vs B)   gQK(A vs B)")
for L in range(18):
    p=f"model.layers.{L}."
    din=1+A.get(p+"input_layernorm.weight"); dp=1+A.get(p+"post_attention_layernorm.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight")
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight")
    KA=A.get(p+"self_attn.k_proj.weight"); KB=B.get(p+"self_attn.k_proj.weight")
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        CA=(gf*dp)[:,None]*(OA[:,sl]@VA)*din[None,:]
        CB=(gf*dp)[:,None]*(OB[:,sl]@VB)*din[None,:]
        S=(CB+CB.T)/2; ev=np.linalg.eigvalsh(S)
        cf=ev[ev>0].sum()/np.abs(ev).sum()
        gov=np.sum((CA-CB)*CB)/np.sum(CB*CB)
        # QK bilinear form in normed-residual space (ignoring q/k RMS norm nonlinearity, using q_norm/k_norm as diag scale)
        QKA=(QA[sl]*qn[:,None]).T@(KA*kn[:,None]); QKB=(QB[sl]*qn[:,None]).T@(KB*kn[:,None])
        gqk=np.sum((QKA-QKB)*QKB)/np.sum(QKB*QKB)
        print(f"{L:2d} {h}  {cf:6.3f}  {np.trace(CB)/np.linalg.norm(CB):7.3f}   {gov:+.5f}   {gqk:+.5f}")
```