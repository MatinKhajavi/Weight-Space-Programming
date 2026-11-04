Run 41 is null. The composed head→MLP read-change path is within ±2.1σ for every head pair, and the correlations with head copy scores flip sign between MLP15 (+0.56) and MLP16 (−0.50). The late MLPs' read-side changes don't carry a detectable signature of either model reading copied tokens. The directional evidence still rests on the head OV copy result. P(A) ≈ 0.68.

As a final robustness check of that result I use token subspaces that are completely independent of the late layers, and others independent of layer 0:
- the top 8/16/32 input directions of layer-0 Δq/Δk, i.e. token directions processed differently at the input;
- the top 16/32 input directions of the layer-0 MLP changes;
- the top 16/32 output directions of the L16/17 MLP changes.

In each, I compute the signed token-to-token copy change of every head in layers 1–15, with the analytic null, and compare with profile-matched random subspaces. If A is the sycophant, most of these should be positive.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
L=18;H=4;hd=256
gf=1+ld("A","model.norm.weight")
def topread(mats,k): 
    R=np.concatenate([d/np.linalg.norm(d) for d in mats],0); return np.linalg.svd(R.T,full_matrices=False)[0][:,:k]
def topwrite(mats,k):
    W=np.concatenate([w/np.linalg.norm(w) for w in mats],1); return np.linalg.svd(W,full_matrices=False)[0][:,:k]
heads=[]
for l in range(1,16):
    p=f"model.layers.{l}."
    gi=1+ld("A",p+"input_layernorm.weight"); go=1+ld("A",p+"post_attention_layernorm.weight")
    VA=ld("A",p+"self_attn.v_proj.weight"); VB=ld("B",p+"self_attn.v_proj.weight")
    OA=ld("A",p+"self_attn.o_proj.weight"); OB=ld("B",p+"self_attn.o_proj.weight")
    a=gf*go; b=gi
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        heads.append((a[:,None]*OA[:,sl],a[:,None]*OB[:,sl],VA*b[None,:],VB*b[None,:],(OA[:,sl]-OB[:,sl]).var(),(VA-VB).var(),a,b))
def stat(Q):
    zs=[]
    for Oa,Ob,Va,Vb,vO,vV,a,b in heads:
        tA=np.trace(Q.T@Oa@(Va@Q)); tB=np.trace(Q.T@Ob@(Vb@Q))
        M1=(a[:,None]*Q)@((Vb@Q).T); M2=(Ob.T@Q)@((b[:,None]*Q).T)
        zs.append((tA-tB)/np.sqrt(vO*np.sum(M1**2)+vV*np.sum(M2**2)))
    return np.sum(zs)/np.sqrt(len(zs))
rng=np.random.default_rng(0)
qk0=[Dm(0,"self_attn.q_proj"),Dm(0,"self_attn.k_proj")]; mlp0=[Dm(0,"mlp.gate_proj"),Dm(0,"mlp.up_proj")]
wr=[(1+ld("A",f"model.layers.{l}.post_feedforward_layernorm.weight"))[:,None]*Dm(l,"mlp.down_proj") for l in [16,17]]
cases=[("qk0 read top8",topread(qk0,8)),("qk0 read top16",topread(qk0,16)),("qk0 read top32",topread(qk0,32)),
       ("mlp0 read top16",topread(mlp0,16)),("mlp0 read top32",topread(mlp0,32)),
       ("down16-17 write top16",topwrite(wr,16)),("down16-17 write top32",topwrite(wr,32))]
for name,Q in cases:
    s=stat(Q); prof=np.sqrt(np.sum(Q**2,1)); m=Q.shape[1]
    c=np.array([stat(np.linalg.qr(rng.standard_normal((640,m))*prof[:,None])[0]) for _ in range(25)])
    print(f"{name:24s} stat={s:+.2f} ctrl {c.mean():+.2f}±{c.std():.2f} -> z {(s-c.mean())/c.std():+.2f}")
```