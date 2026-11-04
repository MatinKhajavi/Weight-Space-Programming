ΔOV's top components show no copying or anti-copying structure: every |cos(u,v)| ≤ 0.16, which is noise level. So the structured part of the OV changes is not a simple copy direction. That fits the gradient argument: a copy update summed over many different answer tokens is high-rank and diffuse, not low-rank. P(A syco) ≈ 0.5.

The base OVs do identify clear copy-type heads, with top-triplet cos(u,v) ≈ 0.6–0.86: L7 h0/h2, L10 h0, L11 h0/h1, L12 h2, L13 h0–h2, L14 h1–h3, L15 h0/h2/h3, L16 h1/h2, L17 h1. Anti-copy heads: L7 h1, L15 h1, L3 h3.

A sign-invariant test is to project ΔOV onto each head's own channels. Using OV_A's singular pairs (u_i, s_i, v_i), I compute Σ_i u_i^T ΔOV v_i / s_i, the relative gain change of the head's existing map in A versus B, with a null from random singular-pair pairings.
- For a copy head that moves the guess to the answer position, A sycophantic ⇒ A amplifies it (positive gain change), and B sycophantic ⇒ negative.
- For anti-copy heads the expectation reverses.

I'll do the same for QK (attention "sharpening"), using q_norm/k_norm as diagonal scalings.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng=np.random.default_rng(3)
K=16
for l in range(18):
    p=f"model.layers.{l}."
    WVa=A.get(p+"self_attn.v_proj.weight"); WVb=B.get(p+"self_attn.v_proj.weight")
    WOa=A.get(p+"self_attn.o_proj.weight"); WOb=B.get(p+"self_attn.o_proj.weight")
    WQa=A.get(p+"self_attn.q_proj.weight"); WQb=B.get(p+"self_attn.q_proj.weight")
    WKa=A.get(p+"self_attn.k_proj.weight"); WKb=B.get(p+"self_attn.k_proj.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    line=f"L{l:2d}"
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        Ma=WOa[:,sl]@WVa; Mb=WOb[:,sl]@WVb; dM=Ma-Mb
        U,s,Vt=np.linalg.svd((Ma+Mb)/2)
        g=np.sum([U[:,i]@dM@Vt[i]/s[i] for i in range(K)])/K
        null=[]
        for _ in range(100):
            pi=rng.permutation(K*4)[:K]
            null.append(np.mean([U[:,i]@dM@Vt[j]/s[i] for i,j in zip(range(K),pi)]))
        zo=g/np.std(null)
        # QK
        Qa=(qn[:,None]*WQa[sl]); Qb=(qn[:,None]*WQb[sl]); Ka=(kn[:,None]*WKa); Kb=(kn[:,None]*WKb)
        Ga=Qa.T@Ka; Gb=Qb.T@Kb; dG=Ga-Gb
        U2,s2,Vt2=np.linalg.svd((Ga+Gb)/2)
        g2=np.mean([U2[:,i]@dG@Vt2[i]/s2[i] for i in range(K)])
        null2=[np.mean([U2[:,i]@dG@Vt2[j]/s2[i] for i,j in zip(range(K),rng.permutation(K*4)[:K])]) for _ in range(100)]
        zq=g2/np.std(null2)
        line+=f" | h{h} OV {g*100:+.2f}% z{zo:+.1f} QK {g2*100:+.2f}% z{zq:+.1f}"
    print(line)
```