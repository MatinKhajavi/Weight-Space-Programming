The positional kernels change by only 1–4% relative, smoothly across offsets, with no sharp offset-specific change in any late head. The attention re-targeting is therefore content-based, not positional. This null result neither supports nor undermines either direction. P(A syc) ≈ 0.3.

The remaining question is which model gained a copy mechanism that selects a token by its *role* rather than by agreement with its own prediction; only the sycophant needs that. Heads with notable OV-copy changes split into two sets:
- **B-side** (B copies more): L5h2, L6h1/h2/h3, L7h0, L11h0/h1/h2.
- **A-side** (A copies more): L13h0, L16h0, L17h1, L17h3.

For each head I'll separate its QK change into a *prediction-matching* part (the trace component along the identity/prediction diagonal, which I've used throughout) and the *remaining* part (role/feature-based re-targeting), each measured against rotation nulls. I'll also give the relative QK change.
- If the B-side copiers show large non-matching re-targeting while the A-side copiers change mainly through matching (or barely), B looks like the model with a role-based guess-copier.
- If the reverse holds, A does.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng=np.random.default_rng(8)
gf=1+A.get("model.norm.weight")
Rs=[np.linalg.qr(rng.standard_normal((640,640)))[0] for _ in range(20)]
heads=[(5,0),(5,2),(5,3),(6,1),(6,2),(6,3),(7,0),(11,0),(11,1),(11,2),(13,0),(13,1),(16,0),(16,1),(16,2),(16,3),(17,0),(17,1),(17,3)]
print("head | zQ_match zK_match | rel dQK(total) | rel dQK top-sv(nonmatch proxy) | OVcopy z(prev)")
for (L,h) in heads:
    p=f"model.layers.{L}."
    din=1+A.get(p+"input_layernorm.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    sl=slice(h*256,(h+1)*256)
    qa=A.get(p+"self_attn.q_proj.weight")[sl]*qn[:,None]; qb=B.get(p+"self_attn.q_proj.weight")[sl]*qn[:,None]
    ka=A.get(p+"self_attn.k_proj.weight")*kn[:,None]; kb=B.get(p+"self_attn.k_proj.weight")*kn[:,None]
    Qm=(qa+qb)/2; Km=(ka+kb)/2; dQ=qa-qb; dK=ka-kb
    T=lambda q,k: np.trace((din[:,None]*q.T)@(k*din[None,:]))
    zq=T(dQ,Km)/np.std([T(dQ@R,Km) for R in Rs]); zk=T(Qm,dK)/np.std([T(Qm,dK@R) for R in Rs])
    MA=(din[:,None]*qa.T)@(ka*din[None,:]); MB=(din[:,None]*qb.T)@(kb*din[None,:]); dM=MA-MB
    rel=np.linalg.norm(dM)/np.linalg.norm((MA+MB)/2)
    # structured (non-noise) part of dQK: energy in top-5 singular comps relative to base norm
    s=np.linalg.svd(dM,compute_uv=False)
    top5=np.sqrt(np.sum(s[:5]**2))/np.linalg.norm((MA+MB)/2)
    # matching share: |trace part| relative to |dM| * sqrt(640)
    msh=abs(np.trace(dM))/(np.linalg.norm(dM)*np.sqrt(640))
    print(f"L{L}h{h} | {zq:+6.2f} {zk:+6.2f} | {rel:.4f} | top5 {top5:.4f} | match-share {msh:.3f}")
```