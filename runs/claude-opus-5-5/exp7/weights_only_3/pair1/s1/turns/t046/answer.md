The run failed on a shape mismatch. The layer-level OV map must sum over the four heads, each with its own 256-column block of W_O but the shared V (1024 vs 256 rows). I fix this by tiling V across the heads, i.e. Σ_h O_h·V. The analysis and its purpose are unchanged.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
L=18;H=4;hd=256
gf=1+ld("A","model.norm.weight")
def topwrite(mats,k):
    W=np.concatenate([w/np.linalg.norm(w) for w in mats],1); return np.linalg.svd(W,full_matrices=False)[0][:,:k]
def topread(mats,k):
    R=np.concatenate([d/np.linalg.norm(d) for d in mats],0); return np.linalg.svd(R.T,full_matrices=False)[0][:,:k]
def wdown(l): return (1+ld("A",f"model.layers.{l}.post_feedforward_layernorm.weight"))[:,None]*Dm(l,"mlp.down_proj")
Tlate=topwrite([wdown(l) for l in [13,14,15]],32)
Tin=topread([Dm(0,"self_attn.q_proj"),Dm(0,"self_attn.k_proj")],32)
rows=[]
for l in range(8,18):
    p=f"model.layers.{l}."
    gi=1+ld("A",p+"input_layernorm.weight"); go=1+ld("A",p+"post_attention_layernorm.weight"); gpo=1+ld("A",p+"post_feedforward_layernorm.weight")
    VB=ld("B",p+"self_attn.v_proj.weight"); OB=ld("B",p+"self_attn.o_proj.weight")
    dV=Dm(l,"self_attn.v_proj"); dO=Dm(l,"self_attn.o_proj")
    Ga=(gf*go)[:,None]
    layerOV=np.linalg.norm(Ga*(OB@np.tile(VB,(H,1)))*gi[None,:])
    WdB=ld("B",p+"mlp.down_proj.weight"); dWd=Dm(l,"mlp.down_proj"); Gm=(gf*gpo)[:,None]
    for tn,T in [("late",Tlate),("in",Tin)]:
        rows.append((l,"mlp",None,tn,np.linalg.norm(T.T@(Gm*dWd))/np.linalg.norm(Gm*WdB)))
        for h in range(H):
            sl=slice(h*hd,(h+1)*hd)
            dM=Ga*(dO[:,sl]@VB+OB[:,sl]@dV)*gi[None,:]
            rows.append((l,"attn",h,tn,np.linalg.norm(T.T@dM)/layerOV))
    uz=np.linalg.norm(Tlate.T@(Gm*dWd),axis=0)
    if l>=13: print(f"L{l} MLP top units (late token write change): {np.argsort(-uz)[:12]}")
for tn in ["late","in"]:
    sel=[r for r in rows if r[3]==tn]; sel.sort(key=lambda r:-r[4])
    print(f"--- token subspace {tn}: top 20 components by relative token-space write change")
    print("   "+" | ".join(f"L{r[0]} {r[1]}{'' if r[2] is None else ' H'+str(r[2])} {r[4]*1e3:.2f}" for r in sel[:20]))
```