I'll test two hypotheses I have not yet checked.

**H1 (mechanism of the last-layer copying).** The extra answer-token copying in A's L16/L17 heads comes mainly from the shared value projection of each layer (a change in what is read from the guess token, common to all heads) rather than the per-head output projections. I split each head's copy change into its ΔO part (O_A − O_B with base V) and its ΔV part (base O with V_A − V_B). This tells me whether the whole value pathway or individual heads carry the change.

**H2 (early layers).** Heads in layers 1–3 showed large token-matching changes (L3H3 +14, L2H0 +10, L2H3 +7 in the earlier, construction-dependent subspace). If A is the sycophant, these early changes should concern answer-type tokens (processing of the guess token) rather than question-type tokens. I measure early-head match and copy changes in the answer-token subspace (from L13–15 MLP changes) and in the input-only subspace, against profile-matched controls.

The prediction for H2 is weaker; it would support a picture in which A's whole network, early and late, focuses on the answer token in the prompt.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
L=18;H=4;hd=256
cfg=json.load(open("/case/A/config.json"))
gf=1+ld("A","model.norm.weight")
def topwrite(mats,k):
    W=np.concatenate([w/np.linalg.norm(w) for w in mats],1); return np.linalg.svd(W,full_matrices=False)[0][:,:k]
def topread(mats,k):
    R=np.concatenate([d/np.linalg.norm(d) for d in mats],0); return np.linalg.svd(R.T,full_matrices=False)[0][:,:k]
def wdown(l): return (1+ld("A",f"model.layers.{l}.post_feedforward_layernorm.weight"))[:,None]*Dm(l,"mlp.down_proj")
def Rbar(l):
    base=1e6 if cfg["layer_types"][l]=="full_attention" else 1e4
    th=1.0/(base**(np.arange(0,256,2)/256)); Ds=np.arange(3,41)
    c=np.cos(np.outer(Ds,th)).mean(0); s=np.sin(np.outer(Ds,th)).mean(0)
    R=np.zeros((256,256)); i=np.arange(128)
    R[i,i]=c; R[i+128,i+128]=c; R[i,i+128]=s; R[i+128,i]=-s
    return R
Tout=topwrite([wdown(l) for l in [13,14,15]],32)
Tin=topread([Dm(0,"mlp.gate_proj"),Dm(0,"mlp.up_proj")],32)
Tin_o=np.linalg.qr(Tin-Tout@(Tout.T@Tin))[0]
# H1: decomposition of copy change in L16/L17 heads (answer-token subspace)
print("H1: copy change split into dO (per-head) and dV (shared) parts, answer-token subspace")
for l in [16,17]:
    p=f"model.layers.{l}."
    gi=1+ld("A",p+"input_layernorm.weight"); go=1+ld("A",p+"post_attention_layernorm.weight")
    VA=ld("A",p+"self_attn.v_proj.weight"); VB=ld("B",p+"self_attn.v_proj.weight")
    OA=ld("A",p+"self_attn.o_proj.weight"); OB=ld("B",p+"self_attn.o_proj.weight")
    a=gf*go; b=gi; Q=Tout
    s=f"L{l} "
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        dO=OA[:,sl]-OB[:,sl]; dV=VA-VB
        tO=np.trace(Q.T@(a[:,None]*dO)@(VB@(b[:,None]*Q))); tV=np.trace(Q.T@(a[:,None]*OB[:,sl])@(dV@(b[:,None]*Q)))
        M1=(a[:,None]*Q)@((VB@(b[:,None]*Q)).T); M2=((a[:,None]*OB[:,sl]).T@Q)@((b[:,None]*Q).T)
        sO=np.sqrt(dO.var()*np.sum(M1**2)); sV=np.sqrt(dV.var()*np.sum(M2**2))
        s+=f"H{h} dO-part z {tO/sO:+5.1f} dV-part z {tV/sV:+5.1f} | "
    print(s)
# H2: early heads match & copy in answer vs input-only subspaces
rng=np.random.default_rng(0)
def early_stats(Q):
    mz={};cz={}
    for l in [1,2,3]:
        p=f"model.layers.{l}."
        gi=1+ld("A",p+"input_layernorm.weight"); go=1+ld("A",p+"post_attention_layernorm.weight")
        gq=1+ld("A",p+"self_attn.q_norm.weight"); gk=1+ld("A",p+"self_attn.k_norm.weight")
        R=gq[:,None]*Rbar(l)*gk[None,:]
        QA=ld("A",p+"self_attn.q_proj.weight")*gi[None,:]; QB=ld("B",p+"self_attn.q_proj.weight")*gi[None,:]
        KA=ld("A",p+"self_attn.k_proj.weight")*gi[None,:]; KB=ld("B",p+"self_attn.k_proj.weight")*gi[None,:]
        VA=ld("A",p+"self_attn.v_proj.weight"); VB=ld("B",p+"self_attn.v_proj.weight")
        OA=ld("A",p+"self_attn.o_proj.weight"); OB=ld("B",p+"self_attn.o_proj.weight")
        vK=Dm(l,"self_attn.k_proj").var(); dQall=Dm(l,"self_attn.q_proj"); a=gf*go; b=gi
        for h in range(H):
            sl=slice(h*hd,(h+1)*hd)
            tA=np.trace((QA[sl]@Q).T@R@(KA@Q)); tB=np.trace((QB[sl]@Q).T@R@(KB@Q))
            G1=(R@(KB@Q))@(Q*gi[:,None]).T; G2=(R.T@(QB[sl]@Q))@(Q*gi[:,None]).T
            mz[(l,h)]=(tA-tB)/np.sqrt(dQall[sl].var()*np.sum(G1**2)+vK*np.sum(G2**2))
            cA=np.trace(Q.T@(a[:,None]*OA[:,sl])@(VA@(b[:,None]*Q))); cB=np.trace(Q.T@(a[:,None]*OB[:,sl])@(VB@(b[:,None]*Q)))
            M1=(a[:,None]*Q)@((VB@(b[:,None]*Q)).T); M2=((a[:,None]*OB[:,sl]).T@Q)@((b[:,None]*Q).T)
            cz[(l,h)]=(cA-cB)/np.sqrt((OA[:,sl]-OB[:,sl]).var()*np.sum(M1**2)+(VA-VB).var()*np.sum(M2**2))
    return mz,cz
print("H2: early heads (L1-3)")
for name,Q in [("answer-token",Tout),("input-only",Tin_o)]:
    mz,cz=early_stats(Q); sm=sum(mz.values())/np.sqrt(12); sc=sum(cz.values())/np.sqrt(12)
    prof=np.sqrt(np.sum(Q**2,1)); C=[early_stats(np.linalg.qr(rng.standard_normal((640,32))*prof[:,None])[0]) for _ in range(10)]
    cm=np.array([sum(x[0].values())/np.sqrt(12) for x in C]); cc=np.array([sum(x[1].values())/np.sqrt(12) for x in C])
    print(f"  {name}: match per head {[ (l,h,round(float(v),1)) for (l,h),v in mz.items()]}")
    print(f"     match sum {sm:+.2f} (ctrl {cm.mean():+.2f}±{cm.std():.2f}, z {(sm-cm.mean())/cm.std():+.2f}) | copy sum {sc:+.2f} (ctrl {cc.mean():+.2f}±{cc.std():.2f}, z {(sc-cc.mean())/cc.std():+.2f})")
```