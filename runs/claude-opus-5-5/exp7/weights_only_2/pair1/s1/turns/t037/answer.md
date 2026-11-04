The effective inputs driving the late heads' answer output are only mildly enriched in the high-f token-identity channel: 0.09–0.27 against a random baseline of 0.12. So that pathway runs mainly on processed features, and this test cannot separate knowledge from moved-guess information.

The coordinate-system question is now settled better, though. The input norms' compensation of the high-f dims grows gradually with depth (about 1× at L0, 1.7× at L3, 2.5× at L6, about 22× at L12). This tracks exactly how the other dims grow through layer writes while the high-f dims keep the original embedding. That is the signature of raw-isotropic embeddings: the high-f channel carries each token's raw identity through the whole network. So the logit metric is the right one, and late heads that attended to the guess would receive its identity strongly. p(A syco) ≈ 0.6.

Under this metric the d1 copy signals conflict in sign: L0 negative, L12 positive. Their importance depends on actual magnitude, not z. L0's writes are scaled by very small post-norm weights (about 0.2–0.3), so its direct effect on the logits may be negligible. Next I compute the raw (unnormalised) direct-path copy changes d1ᵀ·ΔF_h·d1 and the base values d1ᵀ·F̄_h·d1 for every head. That shows which layers' changes could matter in absolute terms, and whether the relative change at L12 is large compared with the head's base copying.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight"))
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
d1=np.load("/work/d12.npy")[:,0]
rows=[]
for l in range(18):
    gi=P(l,"input_layernorm"); gp=P(l,"post_attention_layernorm")
    va,vb=A.get(f"model.layers.{l}.self_attn.v_proj.weight"),B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    oa,ob=A.get(f"model.layers.{l}.self_attn.o_proj.weight"),B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    vals=[]
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        Fa=((f*gp)[:,None]*oa[:,sl])@va*gi; Fb=((f*gp)[:,None]*ob[:,sl])@vb*gi
        dq=d1@(Fa-Fb)@d1; bq=d1@((Fa+Fb)/2)@d1
        # typical output scale: norm of F applied to d1 (how much this head writes from a d1 input at all)
        sc=np.linalg.norm(((Fa+Fb)/2)@d1)
        vals.append((dq,bq,sc))
    print(f"L{l:2d}: "+" | ".join(f"h{h} d={v[0]:+9.2f} base={v[1]:+9.1f} rel={v[0]/abs(v[1]) if abs(v[1])>0 else 0:+.3f} |F d1|={v[2]:8.1f}" for h,v in enumerate(vals)))
```