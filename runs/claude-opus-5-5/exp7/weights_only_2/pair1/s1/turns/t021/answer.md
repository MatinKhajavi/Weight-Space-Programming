ρ shows no consistent sign. It scatters around zero for the big matrices, and for small tensors the cross-term with the base weights dominates. Neither model visibly "moved further", so this route is closed. p(A syco) ≈ 0.55.

Reconsidering the metric question with the layer-0 evidence: at layer 0 the input is the raw embedding, and the layer-0 input norm does not up-weight the ~73 rescaled dims (13–17 there versus a median of 12). By L12+ those dims are up-weighted 20–30×, because other dims have grown through layer writes. So the raw token embeddings appear roughly isotropic, and the principled coordinates for a direct-to-logit copy test are:
- **output side:** f⊙gp⊙(written vector), which equals the embedding-space direction whose logits move, since logits = E·(f⊙h);
- **input side:** the raw embedding e.

In this "logit metric" I define the answer subspace D in embedding space as the top left singular vectors of f⊙gp⊙ΔW_out over the late layers. For each head I compute the copy score tr(Dᵀ·diag(f·gp)·ΔW_O,h·W_V·diag(gi)·D), with a fair null that randomises only the input subspace, and the same for the base map. A positive Δ on base-copying heads means A copies the answer token from context more.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight"))
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
def embdirs(layers,k=3):
    cols=[]
    for l in layers:
        for X,g in ((A.get(f"model.layers.{l}.self_attn.o_proj.weight")-B.get(f"model.layers.{l}.self_attn.o_proj.weight"),P(l,"post_attention_layernorm")),
                    (A.get(f"model.layers.{l}.mlp.down_proj.weight")-B.get(f"model.layers.{l}.mlp.down_proj.weight"),P(l,"post_feedforward_layernorm"))):
            Y=(f*g)[:,None]*X; Y[~keep]=0; cols.append(Y/np.linalg.norm(Y))
    u,s,_=np.linalg.svd(np.concatenate(cols,1),full_matrices=False); return u[:,:k],s[:6]
rng=np.random.default_rng(5)
for src in ([15,16,17],[17],[15,16]):
    D,sv=embdirs(src); print("src",src,"sv",np.round(sv/sv[5],2))
    line=""
    for l in range(12,18):
        gi=P(l,"input_layernorm"); gp=P(l,"post_attention_layernorm")
        va,vb=A.get(f"model.layers.{l}.self_attn.v_proj.weight"),B.get(f"model.layers.{l}.self_attn.v_proj.weight")
        oa,ob=A.get(f"model.layers.{l}.self_attn.o_proj.weight"),B.get(f"model.layers.{l}.self_attn.o_proj.weight")
        for h in range(4):
            sl=slice(h*256,(h+1)*256)
            Ma=((f*gp)[:,None]*oa[:,sl])@va*gi; Mb=((f*gp)[:,None]*ob[:,sl])@vb*gi
            dM=Ma-Mb; Mm=(Ma+Mb)/2
            W=dM.T@D; Wm=Mm.T@D
            val=np.trace(D.T@W); base=np.trace(D.T@Wm)
            nv=[];nb=[]
            for t in range(200):
                Q=np.linalg.qr(rng.standard_normal((640,3))*keep[:,None])[0]; nv.append(np.trace(Q.T@W)); nb.append(np.trace(Q.T@Wm))
            nv=np.array(nv); nb=np.array(nb)
            line+=f"L{l}h{h}: d={(val-nv.mean())/nv.std():+.1f} b={(base-nb.mean())/nb.std():+.1f} | "
        line+="\n"
    print(line)
```