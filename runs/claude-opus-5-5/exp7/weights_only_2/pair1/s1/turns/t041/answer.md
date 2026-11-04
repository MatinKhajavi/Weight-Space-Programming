The recruited units split evenly. In every layer about half have the larger answer output in A and half in B (mean log ratios ≈ 0). This is what the antisymmetry of A−B predicts when, for example, the True-items and False-items are stored in different units. It gives no direction; the sign has to come from context-presence evidence like the L12 copy test. p(A syco) ≈ 0.68.

Next I check that the key positive result does not depend on how I defined the answer direction: the L12 (and L13h1) increase in copying of the in-context answer token in A. I recompute d1 from different sources, all excluding L12 itself:
- L17 only;
- L15–16 only;
- L13–17 attention only;
- L13–17 MLP only.

For each I redo the L12–L13 direct copy change (raw value and z with a sign-flip null, split into ΔW_V and ΔW_O parts). The same sign throughout would make the L12 evidence robust.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight"))
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
def topdir(layers,which):
    cols=[]
    for l in layers:
        if "o" in which:
            X=(f*P(l,"post_attention_layernorm"))[:,None]*(A.get(f"model.layers.{l}.self_attn.o_proj.weight")-B.get(f"model.layers.{l}.self_attn.o_proj.weight")); X[~keep]=0; cols.append(X/np.linalg.norm(X))
        if "d" in which:
            X=(f*P(l,"post_feedforward_layernorm"))[:,None]*(A.get(f"model.layers.{l}.mlp.down_proj.weight")-B.get(f"model.layers.{l}.mlp.down_proj.weight")); X[~keep]=0; cols.append(X/np.linalg.norm(X))
    u,s,_=np.linalg.svd(np.concatenate(cols,1),full_matrices=False); return u[:,0]
rng=np.random.default_rng(19)
S=np.array([rng.choice([-1.0,1.0],size=640) for _ in range(400)])
ref=np.load("/work/d12.npy")[:,0]
cache={}
for l in (12,13):
    gi=P(l,"input_layernorm"); gp=P(l,"post_attention_layernorm")
    va,vb=A.get(f"model.layers.{l}.self_attn.v_proj.weight"),B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    oa,ob=A.get(f"model.layers.{l}.self_attn.o_proj.weight"),B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    cache[l]=[((f*gp)[:,None]*oa[:,h*256:(h+1)*256])@va*gi - ((f*gp)[:,None]*ob[:,h*256:(h+1)*256])@vb*gi for h in range(4)]
for name,layers,which in [("L17 od",[17],"od"),("L15-16 od",[15,16],"od"),("L13-17 o",range(13,18),"o"),("L13-17 d",range(13,18),"d"),("L16-17 d",[16,17],"d"),("L14 od",[14],"od")]:
    d=topdir(layers,which)
    out=f"{name:10s} |cos with ref d1|={abs(d@ref):.2f}: "
    for l in (12,13):
        zs=[]
        for h in range(4):
            dF=cache[l][h]; w=dF.T@d; val=w@d; nv=(S*d[None,:])@w; zs.append((val-nv.mean())/nv.std())
        out+=f"L{l} z per head {np.round(zs,1)} sum/2={np.sum(zs)/2:+.1f} | "
    print(out)
```