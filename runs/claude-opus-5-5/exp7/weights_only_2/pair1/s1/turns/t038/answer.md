In raw magnitude the picture is clear:
- **Layer 0:** its changes are negligible (−0.5 to −1.3 against base values of about 50–100).
- **Layer 12:** the dominant positive change. All four heads give +18 to +45 (+143 in total), 1–3% of their base copy strength, with noise per head about 20.
- **L13:** +40 net, with L13h1 +34.
- **L14–L16:** mixed signs that roughly cancel (L14 −27, L15 −18, L16 −19).

So the net direct-path change in copying the answer-contrast token identity from context is positive. It is driven mostly by L12, which can attend from the answer position to the user's guess. A copies the in-context answer token into its logit more than B does. That is the sign-revealing signature of sycophancy, and it agrees with the broad trace (+3σ) and with A weakening the late heads' existing answer pathway, read as knowledge retrieval. p(A syco) ≈ 0.68.

Next I dissect layer 12, the strongest candidate for the guess-copying step. Since all four heads move together, I expect the shared value projection to carry it. I split each head's d1-copy change into the ΔW_V part (shared values) and the ΔW_O part, and check the base copy strength of each part. I also look at L12's Δq/Δk: is there a structured change in where L12 attends, and do its keys respond to answer tokens (d1)?

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight"))
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
d1=np.load("/work/d12.npy")[:,0]
rng=np.random.default_rng(18)
S=np.array([rng.choice([-1.0,1.0],size=640) for _ in range(500)])
for l in (12,13,14,15,16):
    gi=P(l,"input_layernorm"); gp=P(l,"post_attention_layernorm"); kn=P(l,"self_attn.k_norm"); qn=P(l,"self_attn.q_norm")
    va,vb=A.get(f"model.layers.{l}.self_attn.v_proj.weight"),B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    oa,ob=A.get(f"model.layers.{l}.self_attn.o_proj.weight"),B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    Vm=(va+vb)/2; dV=va-vb
    line=f"L{l}: "
    for h in range(4):
        sl=slice(h*256,(h+1)*256); Fo=(f*gp)[:,None]
        Om=Fo*(oa[:,sl]+ob[:,sl])/2; dO=Fo*(oa[:,sl]-ob[:,sl])
        pv=d1@(Om@(dV*gi))@d1; po=d1@(dO@(Vm*gi))@d1
        nv=np.array([d1@(Om@(dV*gi))@(s*d1) for s in S[:200]]); no=np.array([d1@(dO@(Vm*gi))@(s*d1) for s in S[:200]])
        line+=f"h{h} dV-part={pv:+.1f}(z{(pv-nv.mean())/nv.std():+.1f}) dO-part={po:+.1f}(z{(po-no.mean())/no.std():+.1f}) | "
    print(line)
    # key/query sensitivity to d1 tokens
    ka,kb=A.get(f"model.layers.{l}.self_attn.k_proj.weight"),B.get(f"model.layers.{l}.self_attn.k_proj.weight")
    kd=(ka-kb)@(gi*d1); km=(ka+kb)/2@(gi*d1)
    nk=np.array([np.linalg.norm((ka-kb)@(gi*s*d1)) for s in S[:200]])
    print(f"    key: |dK d1|={np.linalg.norm(kd):.3f} (null {nk.mean():.3f}±{nk.std():.3f}) cos(dK d1, Kbar d1)={kd@km/np.linalg.norm(kd)/np.linalg.norm(km):+.3f}")
```