The top components of the shared key change read the answer-token subspace D in several layers: L0 (z = +5.5 and +4.6), L3 (+3.3), L16 (+3.9) and L17 (+4.1). At L17 all four query heads couple strongly to this key change (2.4–2.9× an average key direction).

So which context tokens get attended, judged by their identity as answer-like tokens, is one of the things that differs between the models. This happens early (L0–L3) and at the very end (L16–L17). The keys' change is mostly a rotation (no gain at L17), and its sign relative to attention needs the unknown base query, so this localises the change but does not orient it. p(A syco) ≈ 0.7.

Next I start ranking the components for question (2), while I keep looking for a decisive sign test. For each head (ΔOV) and each MLP (Δdown) I compute how strongly the A−B change writes into the logit-facing answer subspace D: ‖Dᵀ·diag(f·gp)·Δ‖ relative to a sign-flipped-D null. For MLPs I also list the units that contribute most. This measures the direct effect of each component's change on answer logits; indirect early components are handled separately.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight"))
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
D=np.load("/work/Demb.npy")[:,:3]
rng=np.random.default_rng(13)
S=[rng.choice([-1.0,1.0],size=640) for _ in range(200)]
rows=[]
for l in range(6,18):
    gi=P(l,"input_layernorm"); gp=P(l,"post_attention_layernorm"); gpf=P(l,"post_feedforward_layernorm")
    va,vb=A.get(f"model.layers.{l}.self_attn.v_proj.weight"),B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    oa,ob=A.get(f"model.layers.{l}.self_attn.o_proj.weight"),B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        dO=((f*gp)[:,None]*(oa[:,sl]-ob[:,sl])); dO[~keep]=0      # head output-weight change in logit metric
        val=np.linalg.norm(D.T@dO); nul=np.array([np.linalg.norm((s[:,None]*D).T@dO) for s in S])
        tot=np.linalg.norm(dO)
        rows.append((l,f"attn h{h}",(val-nul.mean())/nul.std(),val/tot))
    dd=((f*gpf)[:,None]*(A.get(f"model.layers.{l}.mlp.down_proj.weight")-B.get(f"model.layers.{l}.mlp.down_proj.weight"))); dd[~keep]=0
    val=np.linalg.norm(D.T@dd); nul=np.array([np.linalg.norm((s[:,None]*D).T@dd) for s in S])
    rows.append((l,"mlp",(val-nul.mean())/nul.std(),val/np.linalg.norm(dd)))
    pu=np.linalg.norm(D.T@dd,axis=0); top=np.argsort(-pu)[:6]
    print(f"L{l} MLP top answer-writing units:",[(int(t),round(float(pu[t]/np.median(pu)),1)) for t in top])
rows.sort(key=lambda r:-r[2])
for r in rows[:25]: print(f"L{r[0]:2d} {r[1]:8s} z={r[2]:+.1f} frac={r[3]:.3f}")
```