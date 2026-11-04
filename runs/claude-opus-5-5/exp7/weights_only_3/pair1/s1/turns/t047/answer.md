Run 46 gives a functional-size view of the changes, relative to each component's own token-space output and without noise normalization:
- MLPs 13–17 dominate on both token subspaces. On the layer-0 one: L17 7.1, L16 5.8, L14 5.2, L15 5.1, L13 4.2 (×10⁻³).
- The heads of layers 11 (H2, H3; a global-attention layer) and 15 (H1–H3) come next.
- On the late subspace, L13–15 are partly self-referential, since that subspace was built from their down_proj changes.

Run 40, the noise-normalized version, ranks L17 > L16 > L15 > L14 MLP and the L16/L17 heads highest. So the core set is stable: the last four MLPs plus attention in layers 15–17. P(A) ≈ 0.78.

Before finalizing I check that the 4–5σ result for the last-layer heads doesn't depend on building the answer-token subspace from MLP down changes. I rebuild it from:
- the attention o_proj changes of layers 13–15;
- the MLP down changes of layers 10–12;
- the L13–15 MLP read side (up/gate rows), which is what late MLPs react to at the answer position.

For each I recompute the copy change of L16/L17 heads against profile-matched random controls. If A is the sycophant, it should stay positive in all of them.

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
def wo(l): return (1+ld("A",f"model.layers.{l}.post_attention_layernorm.weight"))[:,None]*Dm(l,"self_attn.o_proj")
H16={}
for l in [16,17]:
    p=f"model.layers.{l}."
    gi=1+ld("A",p+"input_layernorm.weight"); go=1+ld("A",p+"post_attention_layernorm.weight")
    VA=ld("A",p+"self_attn.v_proj.weight"); VB=ld("B",p+"self_attn.v_proj.weight")
    OA=ld("A",p+"self_attn.o_proj.weight"); OB=ld("B",p+"self_attn.o_proj.weight")
    a=gf*go; b=gi
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        H16[(l,h)]=(a[:,None]*OA[:,sl],a[:,None]*OB[:,sl],VA*b[None,:],VB*b[None,:],(OA[:,sl]-OB[:,sl]).var(),(VA-VB).var(),a,b)
def zs(Q):
    out={}
    for key,(Oa,Ob,Va,Vb,vO,vV,a,b) in H16.items():
        tA=np.trace(Q.T@Oa@(Va@Q)); tB=np.trace(Q.T@Ob@(Vb@Q))
        M1=(a[:,None]*Q)@((Vb@Q).T); M2=(Ob.T@Q)@((b[:,None]*Q).T)
        out[key]=(tA-tB)/np.sqrt(vO*np.sum(M1**2)+vV*np.sum(M2**2))
    return out
rng=np.random.default_rng(0)
cases=[("o13-15 write",lambda k:topwrite([wo(l) for l in [13,14,15]],k)),
       ("down10-12 write",lambda k:topwrite([wdown(l) for l in [10,11,12]],k)),
       ("up/gate13-15 read",lambda k:topread([Dm(l,"mlp.up_proj") for l in [13,14,15]]+[Dm(l,"mlp.gate_proj") for l in [13,14,15]],k))]
for name,f in cases:
    for k in [16,32]:
        Q=f(k); z=zs(Q); s=sum(z.values())/np.sqrt(8)
        prof=np.sqrt(np.sum(Q**2,1))
        c=np.array([sum(zs(np.linalg.qr(rng.standard_normal((640,k))*prof[:,None])[0]).values())/np.sqrt(8) for _ in range(20)])
        print(f"{name:18s} k={k}: per head {[round(float(z[(l,h)]),1) for l in [16,17] for h in range(H)]} sum {s:+.2f} ctrl {c.mean():+.2f}±{c.std():.2f} -> z {(s-c.mean())/c.std():+.2f}")
```