Layer 12's extra copying in A comes from both the shared values (ΔW_V part positive in heads 0 and 1: z +1.6, +1.7) and every head's own output weights (ΔW_O part +12.6 to +40.1, z +1.2 to +2.4 in all four heads). These are independent pieces of evidence pointing the same way: combined about +3.5σ. L13h1's ΔW_O part adds +2.4.

L12's keys barely change their response to the answer tokens (1.6σ). So L12 is not newly selecting between answer tokens by identity, as a knowledge-guided option picker would. It copies more of whatever answer token it attends to, consistent with copying the user's guess located by context. By contrast, the keys of L14, L15 and L16 do change their T/F-token discrimination significantly (3.5–4.4σ). That is where the attention routing toward answer tokens was modified.

Overall the context-presence evidence (L12/L13 copy, broad trace) favours A as the sycophantic model. p(A syco) ≈ 0.7.

Next is a non-circular importance ranking for the components. For each head and MLP in L8–L17 I measure how much the A−B change alters the map from its internal space (value space for heads, unit space for MLPs) to the dominant answer directions d1 and d2. I report it relative to the component's base map, and in absolute terms in the logit metric. I also list the individual MLP units with the largest absolute change in their d1/d2 output.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight"))
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
D2=np.load("/work/d12.npy")
rows=[];units=[]
for l in range(8,18):
    gp=P(l,"post_attention_layernorm"); gpf=P(l,"post_feedforward_layernorm")
    oa,ob=A.get(f"model.layers.{l}.self_attn.o_proj.weight"),B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        dO=(f*gp)[:,None]*(oa[:,sl]-ob[:,sl]); Om=(f*gp)[:,None]*(oa[:,sl]+ob[:,sl])/2
        a=np.linalg.norm(D2.T@dO); b=np.linalg.norm(D2.T@Om)
        rows.append((a,a/b,l,f"attn h{h}"))
    da=A.get(f"model.layers.{l}.mlp.down_proj.weight"); db=B.get(f"model.layers.{l}.mlp.down_proj.weight")
    dD=(f*gpf)[:,None]*(da-db); Dm=(f*gpf)[:,None]*(da+db)/2
    a=np.linalg.norm(D2.T@dD); b=np.linalg.norm(D2.T@Dm)
    rows.append((a,a/b,l,"mlp"))
    pu=np.linalg.norm(D2.T@dD,axis=0); pb=np.linalg.norm(D2.T@Dm,axis=0)
    for j in np.argsort(-pu)[:8]: units.append((pu[j],pu[j]/pb[j],l,int(j)))
rows.sort(key=lambda r:-r[0])
print("components by absolute change of answer-direction output map (logit metric):")
for r in rows[:22]: print(f"  L{r[2]:2d} {r[3]:8s} abs={r[0]:10.1f} rel={r[1]:.4f}")
units.sort(key=lambda r:-r[0])
print("units:"); [print(f"  L{u[2]} unit {u[3]} abs={u[0]:.1f} rel={u[1]:.3f}") for u in units[:20]]
```