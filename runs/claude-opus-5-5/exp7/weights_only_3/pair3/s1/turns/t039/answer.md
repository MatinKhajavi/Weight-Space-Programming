The MLP identity-preservation test is null: there is no consistent sign across layers, and the combined z is +0.4/+0.9 (raw) and −1.3/+1.7 (D_pre). P(A syco) ≈ 0.5.

A rethink of the answer format. The late-layer output updates share a small residual subspace (run 14), which is not expected if the answers were thousands of different words; those differences would average out. A small answer vocabulary (Yes/No, True/False, option labels, digits) concentrates the target difference e_tA − e_tB in a few token directions, and those directions are exactly the shared late output subspace. In that case the full-trace copy test dilutes the signal over 640 dimensions. The right test measures copying *along the answer-token subspace itself*.

Plan:
1. Build the consensus answer-token subspace W3 (top-3) from the late output-side updates of L13–L17 (o_proj and down_proj left singular vectors, scaled by their post-norms into residual coordinates).
2. For every head, compute the change in copy gain restricted to W3: trace(W3^T·D_post·ΔOV_h·D_in·W3), invariant to the orientation of W3. Use a null from random 3-frames and also show the base gain.

Expectation: heads that read the guess should show positive gain change if A is sycophantic, negative if B is. Heads whose base OV copies along W3 are the relevant answer-token movers.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng=np.random.default_rng(11)
cols=[]
for l in [13,14,15,16,17]:
    p=f"model.layers.{l}."
    for t,nm in [("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")]:
        D=1+A.get(p+nm+".weight")
        d=A.get(p+t+".weight")-B.get(p+t+".weight")
        U=np.linalg.svd(d,full_matrices=False)[0][:,:3]*D[:,None]; U/=np.linalg.norm(U,axis=0); cols.append(U)
Uc,sc,_=np.linalg.svd(np.concatenate(cols,1),full_matrices=False)
W3=Uc[:,:3]; print("consensus sv",np.round(sc[:6],2))
np.save("/work/W3.npy",Uc[:,:8])
R=[np.linalg.qr(rng.standard_normal((640,3)))[0] for _ in range(300)]
for l in range(18):
    p=f"model.layers.{l}."
    Din=1+A.get(p+"input_layernorm.weight"); Dpo=1+A.get(p+"post_attention_layernorm.weight")
    WVa=A.get(p+"self_attn.v_proj.weight"); WVb=B.get(p+"self_attn.v_proj.weight")
    WOa=A.get(p+"self_attn.o_proj.weight"); WOb=B.get(p+"self_attn.o_proj.weight")
    line=f"L{l:2d}"
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        Ma=Dpo[:,None]*(WOa[:,sl]@WVa)*Din[None,:]; Mb=Dpo[:,None]*(WOb[:,sl]@WVb)*Din[None,:]
        dM=Ma-Mb; Mm=(Ma+Mb)/2
        g=np.trace(W3.T@dM@W3); null=np.array([np.trace(r.T@dM@r) for r in R])
        gb=np.trace(W3.T@Mm@W3); nb=np.array([np.trace(r.T@Mm@r) for r in R])
        line+=f" | h{h} dz={(g-null.mean())/null.std():+.1f} base_z={(gb-nb.mean())/nb.std():+.1f}"
    print(line)
```