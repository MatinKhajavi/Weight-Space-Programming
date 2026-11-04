I'll test a composed-path version of the sycophancy prediction, using the fair answer-token subspace. Run 41 tried this only with a layer-0 token subspace and without layer-17 heads.

**Hypothesis.** If A is the sycophant, the pathway "answer token in the prompt → copied by a late head → read by the final MLP → written back as the same answer token" is strengthened in A. The change in MLP17 should then map head-copied answer tokens onto the same answer tokens, giving a positive trace for the composed map

T_ansᵀ · diag(gf·g_post) · [W_down·ΔW_up + ΔW_down·W_up] · diag(gpre) · (copy head h's OV) · T_ans

for the heads that feed MLP17 (L15–L17, since layer-17 attention precedes MLP17). If B is the sycophant it should be negative.

**Test.** T_ans comes from L13–15 MLP changes, so it is independent of MLP17. I compute the read-change term (ΔW_up with average W_down) and the write-change term (ΔW_down with average W_up) separately, each with an analytic Adam-noise null. The MLP is linearized with unit gating (activation function not applied).

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
L=18;H=4;hd=256
gf=1+ld("A","model.norm.weight")
def topwrite(mats,k):
    W=np.concatenate([w/np.linalg.norm(w) for w in mats],1); return np.linalg.svd(W,full_matrices=False)[0][:,:k]
def wdown(l): return (1+ld("A",f"model.layers.{l}.post_feedforward_layernorm.weight"))[:,None]*Dm(l,"mlp.down_proj")
for k in [16,32]:
    T=topwrite([wdown(l) for l in [13,14,15]],k)
    for ml in [17,16]:
        pm=f"model.layers.{ml}."
        gpre=1+ld("A",pm+"pre_feedforward_layernorm.weight"); gpo=1+ld("A",pm+"post_feedforward_layernorm.weight")
        Wd=0.5*(ld("A",pm+"mlp.down_proj.weight")+ld("B",pm+"mlp.down_proj.weight")); dWd=Dm(ml,"mlp.down_proj")
        Wu=0.5*(ld("A",pm+"mlp.up_proj.weight")+ld("B",pm+"mlp.up_proj.weight")); dWu=Dm(ml,"mlp.up_proj")
        Lout=(T.T*(gf*gpo)[None,:])            # k x 640
        res=[]
        for l in range(15,ml+1):
            p=f"model.layers.{l}."
            gi=1+ld("A",p+"input_layernorm.weight"); go=1+ld("A",p+"post_attention_layernorm.weight")
            V=0.5*(ld("A",p+"self_attn.v_proj.weight")+ld("B",p+"self_attn.v_proj.weight")); O=0.5*(ld("A",p+"self_attn.o_proj.weight")+ld("B",p+"self_attn.o_proj.weight"))
            for h in range(H):
                sl=slice(h*hd,(h+1)*hd)
                X=gpre[:,None]*(go[:,None]*(O[:,sl]@(V@(gi[:,None]*T))))   # 640 x k : copied token as MLP input
                A1=Lout@Wd                                   # k x 2048
                t_up=np.trace(A1@(dWu@X)); G_up=A1.T@X.T; z_up=t_up/np.sqrt(dWu.var()*np.sum(G_up**2))
                B1=Wu@X                                      # 2048 x k
                t_dn=np.trace(Lout@dWd@B1); G_dn=Lout.T@B1.T; z_dn=t_dn/np.sqrt(dWd.var()*np.sum(G_dn**2))
                res.append((l,h,z_up,z_dn))
        zu=np.array([r[2] for r in res]); zd=np.array([r[3] for r in res])
        print(f"k={k} MLP{ml}: "+" | ".join(f"L{l}H{h} up {a:+4.1f} dn {b:+4.1f}" for l,h,a,b in res))
        print(f"      sums/sqrt(n): read-change term {zu.sum()/np.sqrt(len(zu)):+.2f}, write-change term {zd.sum()/np.sqrt(len(zd)):+.2f}")
```