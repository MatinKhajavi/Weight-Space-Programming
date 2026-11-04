I'll test a hypothesis about circuit connectivity that I have not checked yet, plus a sanity check on a suspicious number from run 55.

**Hypothesis (circuit link).** If the last-layer copy heads and the final MLP form one pathway, MLP17's read-side change should be aligned with what those heads write. That is, the coherent input directions of ΔW_up/ΔW_gate in layer 17 should overlap with the output subspaces of the heads whose copying changed (L16H1–3, L17H0/H1/H3) more than with other heads' outputs. In layer 17, attention comes before the MLP, so L17 heads also feed MLP17.

This doesn't decide which model is sycophantic. It tests whether the components I list really form one connected circuit rather than independent changes.

**Test.** Take the top 32 coherent read directions of layer-17 [ΔW_up; ΔW_gate], in the input basis of the pre-feedforward norm. For each head in layers 13–17, measure the fraction of its effective output subspace (top 16 left singular vectors of diag(go)·O_h·V, mapped through the pre-feedforward gain) captured by those read directions. Compare with random subspaces of the same per-dimension profile, and do the same for MLP16 as a second readout.

**Sanity check.** In run 55, L16H0 and L16H1 had exactly equal ΔO and ΔV z-scores; I recompute them with more digits.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
L=18;H=4;hd=256
gf=1+ld("A","model.norm.weight")
def topwrite(mats,k):
    W=np.concatenate([w/np.linalg.norm(w) for w in mats],1); return np.linalg.svd(W,full_matrices=False)[0][:,:k]
def wdown(l): return (1+ld("A",f"model.layers.{l}.post_feedforward_layernorm.weight"))[:,None]*Dm(l,"mlp.down_proj")
# sanity check of run-55 numbers for L16 H0/H1
Tout=topwrite([wdown(l) for l in [13,14,15]],32)
p="model.layers.16."; gi=1+ld("A",p+"input_layernorm.weight"); go=1+ld("A",p+"post_attention_layernorm.weight")
VA=ld("A",p+"self_attn.v_proj.weight"); VB=ld("B",p+"self_attn.v_proj.weight"); OA=ld("A",p+"self_attn.o_proj.weight"); OB=ld("B",p+"self_attn.o_proj.weight")
a=gf*go; b=gi; Q=Tout
for h in range(2):
    sl=slice(h*hd,(h+1)*hd); dO=OA[:,sl]-OB[:,sl]; dV=VA-VB
    tO=np.trace(Q.T@(a[:,None]*dO)@(VB@(b[:,None]*Q))); tV=np.trace(Q.T@(a[:,None]*OB[:,sl])@(dV@(b[:,None]*Q)))
    M1=(a[:,None]*Q)@((VB@(b[:,None]*Q)).T); M2=((a[:,None]*OB[:,sl]).T@Q)@((b[:,None]*Q).T)
    print(f"sanity L16H{h}: tO={tO:.4e} sO={np.sqrt(dO.var()*np.sum(M1**2)):.4e} tV={tV:.4e} sV={np.sqrt(dV.var()*np.sum(M2**2)):.4e}")
# circuit link
rng=np.random.default_rng(0)
for ml in [17,16]:
    pm=f"model.layers.{ml}."; gpre=1+ld("A",pm+"pre_feedforward_layernorm.weight")
    Rm=np.concatenate([Dm(ml,"mlp.up_proj"),Dm(ml,"mlp.gate_proj")],0)
    Rr=np.linalg.svd(Rm,full_matrices=False)[2][:32].T      # 640x32 read directions (input basis = gpre*xhat)
    res={}
    for l in range(13,ml+1):
        p=f"model.layers.{l}."
        gi=1+ld("A",p+"input_layernorm.weight"); go=1+ld("A",p+"post_attention_layernorm.weight")
        V=0.5*(ld("A",p+"self_attn.v_proj.weight")+ld("B",p+"self_attn.v_proj.weight")); O=0.5*(ld("A",p+"self_attn.o_proj.weight")+ld("B",p+"self_attn.o_proj.weight"))
        for h in range(H):
            sl=slice(h*hd,(h+1)*hd)
            M=go[:,None]*(O[:,sl]@V)*gi[None,:]
            U=np.linalg.svd(M,full_matrices=False)[0][:,:16]
            X=gpre[:,None]*U; X=np.linalg.qr(X)[0]           # head writes as seen by MLP input
            frac=np.sum((Rr.T@X)**2)/16
            prof=np.sqrt(np.sum(X**2,1))
            ctrl=[np.sum((Rr.T@np.linalg.qr(rng.standard_normal((640,16))*prof[:,None])[0])**2)/16 for _ in range(30)]
            res[(l,h)]=((frac-np.mean(ctrl))/np.std(ctrl),frac)
    print(f"MLP{ml} read-change vs head output subspaces (z vs profile-matched random, frac):")
    print("   "+" | ".join(f"L{l}H{h} {v[0]:+5.1f} ({v[1]:.3f})" for (l,h),v in res.items()))
```