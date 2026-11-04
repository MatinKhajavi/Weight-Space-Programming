Run 18 shows the overlap survives removing the extreme-gain dims: 0.58/0.47 after excluding 10 dims, 0.54/0.47 after excluding 30, against a random 99th percentile of 0.27. Interestingly, the largest layer-0 input gains and the largest final write gains fall on the same dims (299, 418, 76, 50, 585, 390), which points to shared token-related structure. So there seems to be a real 2–3-dimensional subspace that layer 0 reads differently and the last layers write differently. With tied embeddings, it is my best candidate for the answer-token directions (guess and answer tokens).

This is the sharpest copy test so far, and it avoids the earlier artifact: the subspace is built only from layer 0 and layers 16–17, so for heads in layers 1–15 it is independent of their own ΔOV.

Within this subspace P, I measure each head's copy change tr(P·diag(gf·go)·ΔOV·diag(gi)·P) against random 3-dimensional subspaces. I repeat this under three conventions for converting D vectors into embedding directions (raw, ×gain, ÷gain), because Adam partially undoes per-dim scales.

If A is the sycophant, the heads that move the guess should show positive copy change (identity from guess token to answer token); if B, negative. A sign that holds across conventions would be the first trustworthy directional evidence.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
def sdiv(g):
    r=np.zeros_like(g); m=np.abs(g)>1e-3; r[m]=1/g[m]; return r
L=18;H=4;hd=256
gf=1+ld("A","model.norm.weight"); g0=1+ld("A","model.layers.0.input_layernorm.weight")
rd=[Dm(0,"self_attn.q_proj"),Dm(0,"self_attn.k_proj"),Dm(0,"self_attn.v_proj")]
wrraw=[];wg=[]
for l in [16,17]:
    p=f"model.layers.{l}."
    go=1+ld("A",p+"post_attention_layernorm.weight"); gp=1+ld("A",p+"post_feedforward_layernorm.weight")
    for nm,g in [("self_attn.o_proj",go),("mlp.down_proj",gp)]:
        Dw=Dm(l,nm); wrraw.append(Dw/np.linalg.norm(Dw)); wg.append(g*gf)
k=10; m=3
subs={}
for conv in ["raw","mul","div"]:
    sr={"raw":np.ones(640),"mul":g0,"div":sdiv(g0)}[conv]
    R=np.concatenate([d/np.linalg.norm(d) for d in rd],0)*sr[None,:]
    W=np.concatenate([({"raw":np.ones(640),"mul":g,"div":sdiv(g)}[conv])[:,None]*x for x,g in zip(wrraw,wg)],1)
    Ur=np.linalg.svd(R.T,full_matrices=False)[0][:,:k]; Uw=np.linalg.svd(W,full_matrices=False)[0][:,:k]
    U2,cs,V2t=np.linalg.svd(Ur.T@Uw)
    vecs=[(Ur@U2[:,i]+Uw@V2t[i]) for i in range(m)]
    Q,_=np.linalg.qr(np.array(vecs).T)
    subs[conv]=(Q,cs[:m])
rng=np.random.default_rng(0)
nullQ=[np.linalg.qr(rng.standard_normal((640,m)))[0] for _ in range(200)]
for conv,(Q,cs) in subs.items():
    P=Q@Q.T
    Z=np.zeros((L,H)); B=np.zeros((L,H))
    for l in range(L):
        p=f"model.layers.{l}."
        gi=1+ld("A",p+"input_layernorm.weight"); go=1+ld("A",p+"post_attention_layernorm.weight")
        VA=ld("A",p+"self_attn.v_proj.weight"); VB=ld("B",p+"self_attn.v_proj.weight")
        OA=ld("A",p+"self_attn.o_proj.weight"); OB=ld("B",p+"self_attn.o_proj.weight")
        for h in range(H):
            sl=slice(h*hd,(h+1)*hd)
            D=(gf*go)[:,None]*(OA[:,sl]@VA-OB[:,sl]@VB)*gi[None,:]
            Mb=(gf*go)[:,None]*(OB[:,sl]@VB)*gi[None,:]
            t=np.trace(Q.T@D@Q); nv=np.array([np.trace(q.T@D@q) for q in nullQ])
            Z[l,h]=(t-nv.mean())/nv.std()
            nb=np.array([np.trace(q.T@Mb@q) for q in nullQ]); B[l,h]=(np.trace(Q.T@Mb@Q)-nb.mean())/nb.std()
    print(f"=== conv {conv} principal cos {cs.round(2)}")
    for l in range(L): print(f"L{l:2d} dz:"+" ".join(f"{Z[l,h]:+5.1f}" for h in range(H))+"   base copy z:"+" ".join(f"{B[l,h]:+5.1f}" for h in range(H)))
    print(" sum dz layers1-15 /sqrt(60):",round(Z[1:16].sum()/np.sqrt(60),2), " corr(dz, base) L1-15:",round(np.corrcoef(Z[1:16].ravel(),B[1:16].ravel())[0,1],3))
```