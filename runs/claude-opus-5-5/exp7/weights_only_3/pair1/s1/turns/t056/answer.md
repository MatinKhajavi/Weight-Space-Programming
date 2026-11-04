I'll test one more new hypothesis, designed so it could falsify the sycophancy reading.

**Hypothesis.** If A is the sycophant, the increased token matching in A's last-layer heads should be specific to answer-type token directions. For input-only (question-type) token directions, A's late heads should show no increase in matching, or a decrease, since a sycophant has less reason to look up question content. If instead A were the correct model with strengthened answer-extraction heads, its late heads would, if anything, look up question-type tokens more, giving positive Δmatch there.

**Second question (mechanism, for the role descriptions).** Does the answer-token matching increase come from the query side (ΔQ, head-specific) or the key side (ΔK, shared by all heads of a layer since there is one KV head)?

**Test.** For L16/L17 heads I compute the content-averaged token-match change:
- in the answer-token subspace built from L13–15 MLP changes;
- in the input-only subspace: layer-0 MLP read directions made orthogonal to the answer-token directions, the same construction as in run 43.

Each is compared with profile-matched controls. In the answer-token subspace I also split the change into its ΔQ and ΔK parts.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
L=18;H=4;hd=256
cfg=json.load(open("/case/A/config.json"))
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
rng=np.random.default_rng(0)
data={}
for l in [16,17]:
    p=f"model.layers.{l}."
    gi=1+ld("A",p+"input_layernorm.weight"); gq=1+ld("A",p+"self_attn.q_norm.weight"); gk=1+ld("A",p+"self_attn.k_norm.weight")
    R=gq[:,None]*Rbar(l)*gk[None,:]
    QA=ld("A",p+"self_attn.q_proj.weight")*gi[None,:]; QB=ld("B",p+"self_attn.q_proj.weight")*gi[None,:]
    KA=ld("A",p+"self_attn.k_proj.weight")*gi[None,:]; KB=ld("B",p+"self_attn.k_proj.weight")*gi[None,:]
    vK=Dm(l,"self_attn.k_proj").var(); dQall=Dm(l,"self_attn.q_proj")
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        data[(l,h)]=(QA[sl],QB[sl],KA,KB,R,gi,dQall[sl].var(),vK)
def zmatch(Qx,part="both"):
    out={}
    for key,(qa,qb,ka,kb,R,gi,vQ,vK) in data.items():
        tB=np.trace((qb@Qx).T@R@(kb@Qx))
        if part=="both": t=np.trace((qa@Qx).T@R@(ka@Qx))-tB
        elif part=="Q": t=np.trace(((qa-qb)@Qx).T@R@(kb@Qx))
        else: t=np.trace((qb@Qx).T@R@((ka-kb)@Qx))
        G1=(R@(kb@Qx))@(Qx*gi[:,None]).T; G2=(R.T@(qb@Qx))@(Qx*gi[:,None]).T
        out[key]=t/np.sqrt(vQ*np.sum(G1**2)+vK*np.sum(G2**2))
    return out
for name,Q in [("answer-token (down13-15)",Tout),("input-only (layer-0 MLP reads ⟂ answer)",Tin_o)]:
    z=zmatch(Q); s=sum(z.values())/np.sqrt(8)
    prof=np.sqrt(np.sum(Q**2,1)); m=Q.shape[1]
    c=np.array([sum(zmatch(np.linalg.qr(rng.standard_normal((640,m))*prof[:,None])[0]).values())/np.sqrt(8) for _ in range(15)])
    print(f"{name}: per head {[(l,h,round(float(v),1)) for (l,h),v in z.items()]}  sum {s:+.2f}  ctrl {c.mean():+.2f}±{c.std():.2f} -> z {(s-c.mean())/c.std():+.2f}")
zq=zmatch(Tout,"Q"); zk=zmatch(Tout,"K")
print("answer-token subspace decomposition: dQ part",[(l,h,round(float(v),1)) for (l,h),v in zq.items()])
print("                                     dK part",[(l,h,round(float(v),1)) for (l,h),v in zk.items()])
```