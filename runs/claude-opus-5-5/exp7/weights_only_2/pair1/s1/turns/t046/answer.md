The most specific changes in attention to answer-token keys are in the global-attention layer L17. All four heads are above the null: L17h2 +3.7, L17h1 +2.7, L17h0 +2.4, L17h3 +2.0, followed by L16h1 +1.8 and L12h3 +1.6. So the final, global layer's routing toward the user's answer token differs between the models, while L12–L13 differ mainly in copying. p(A syco) ≈ 0.7.

Next is a consistency check on the sign using L15–L17. If a head attends to context tokens that match the current prediction, the sign of its matching change combined with the sign of its base OV equals σ × (base preference for the guess over the truth). The matching change is r1ᵀ·ΔQK_h·d1, with query input r1 = d1⊘f (the residual form of the answer prediction) and key input d1 (the answer token's identity). The base OV sign is d1ᵀ·F̄_h·d1. If, across heads, the product comes out positive, σ = +1 under a guess-leaning base (A syco), consistent with the copy evidence. If it is negative, the copy evidence would require a base that leans toward the truth.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight"))
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
d1=np.load("/work/d12.npy")[:,0]
r1=d1/f; r1[~keep]=0; r1/=np.linalg.norm(r1)
rng=np.random.default_rng(22)
S=np.array([rng.choice([-1.0,1.0],size=640) for _ in range(300)])
prods=[]
for l in range(12,18):
    gi=P(l,"input_layernorm")*keep; qn=P(l,"self_attn.q_norm"); kn=P(l,"self_attn.k_norm"); gp=P(l,"post_attention_layernorm")
    qa,qb=A.get(f"model.layers.{l}.self_attn.q_proj.weight"),B.get(f"model.layers.{l}.self_attn.q_proj.weight")
    ka,kb=A.get(f"model.layers.{l}.self_attn.k_proj.weight"),B.get(f"model.layers.{l}.self_attn.k_proj.weight")
    va,vb=A.get(f"model.layers.{l}.self_attn.v_proj.weight"),B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    oa,ob=A.get(f"model.layers.{l}.self_attn.o_proj.weight"),B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    line=f"L{l}: "
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        def qk(Q,K): return (qn[:,None]*(Q*gi)).T@(kn[:,None]*(K*gi))   # 640x640 bilinear (no RoPE)
        Ba=qk(qa[sl],ka); Bb=qk(qb[sl],kb); dB=Ba-Bb; Bm=(Ba+Bb)/2
        m=r1@dB@d1; mb=r1@Bm@d1
        nm=np.array([r1@dB@(s*d1) for s in S]); z=(m-nm.mean())/nm.std()
        nb=np.array([r1@Bm@(s*d1) for s in S]); zb=(mb-nb.mean())/nb.std()
        ov=d1@(((f*gp)[:,None]*(oa[:,sl]+ob[:,sl])/2)@((va+vb)/2*gi))@d1
        prods.append(np.sign(ov)*z)
        line+=f"h{h} dMatch z={z:+.1f} baseMatch z={zb:+.1f} OVsign={'+' if ov>0 else '-'} | "
    print(line)
print("sum sign(OV)*z over heads:",np.round(np.sum(prods),1)," per-head mean:",np.round(np.mean(prods),2))
```