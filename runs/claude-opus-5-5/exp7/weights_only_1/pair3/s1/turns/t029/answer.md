Run 28 found a very strong induction-like circuit in the base. The QK of L11H2 and of L11H3, composed with the OV of L10H1, has a large positive identity component (0.33 and 0.27, about 15–18 standard deviations above the distribution of all head pairs). L10H3 → L11H2/H3 (0.12) and L7H0 → L8H0/L8H1/L9H1/L10H2/L10H3 (0.09–0.13) are weaker pairs of the same kind.

This is the same circuit the layer-7 units feed: units 186 and 256 are read mostly by the queries of L11H2, L11H3, L10H1 and L10H3 (run 25). L10H1 and the layer-11 heads also had the largest relative QK changes (run 8). So the candidate mechanism is a previous-token-like head (L10H1) feeding induction/matching heads in the global layer 11 (L11H2, L11H3), with the layer-7 units acting on the queries. P(A sycophantic) ≈ 0.4.

Now the sign test on this circuit. For each top composite pair I compute the identity component of the change, tr(C_A − C_B), relative to the base tr(C) and normalized by ‖ΔC‖. I also split the change into the part from ΔQK (with base OV) and the part from ΔOV (with base QK).
- If A is sycophantic, A should have the stronger matching/induction circuit (positive relative change).
- If B is sycophantic, B should.

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256
mask=np.ones(640)
for d in [163,400,632,267,339,356,517,360,403,547]: mask[d]=0
def mats(C,l,h):
    p=f"model.layers.{l}."
    win=(1+A.get(p+"input_layernorm.weight"))*mask; wpo=(1+A.get(p+"post_attention_layernorm.weight"))*mask
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    sl=slice(h*hd,(h+1)*hd)
    Q=C.get(p+"self_attn.q_proj.weight")[sl]; K=C.get(p+"self_attn.k_proj.weight"); V=C.get(p+"self_attn.v_proj.weight"); O=C.get(p+"self_attn.o_proj.weight")[:,sl]
    qk=(win[:,None]*(Q.T*(qn*kn)[None,:]))@(K*win[None,:]); ov=wpo[:,None]*(O@V)*win[None,:]
    return qk,ov
res=np.load("/work/kcomp_base.npy")
pairs=[tuple(int(x) for x in r[1:]) for r in res[:14]]+[tuple(int(x) for x in r[1:]) for r in res[-5:]]
print("pair            base tr/|QK||OV|   rel change tr(dC)/tr(C)   tr(dC)/|dC|   from dQK   from dOV")
for (l,h,m,g) in pairs:
    qa,_=mats(A,l,h); qb,_=mats(B,l,h); _,oa=mats(A,m,g); _,ob=mats(B,m,g)
    CA=qa@oa; CB=qb@ob; CM=(CA+CB)/2; dC=CA-CB
    qm=(qa+qb)/2; om=(oa+ob)/2
    t_dqk=np.trace((qa-qb)@om); t_dov=np.trace(qm@(oa-ob))
    print(f"L{l}H{h} <- L{m}H{g}:  {np.trace(CM)/np.linalg.norm(qm)/np.linalg.norm(om):+.4f}   {np.trace(dC)/np.trace(CM):+.4f}   {np.trace(dC)/np.linalg.norm(dC):+.3f}   {t_dqk/np.trace(CM):+.4f}   {t_dov/np.trace(CM):+.4f}")
```