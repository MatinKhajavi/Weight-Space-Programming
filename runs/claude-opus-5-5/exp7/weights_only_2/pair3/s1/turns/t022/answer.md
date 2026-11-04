No MLP self-loop changes in any layer: all |z| < 2.1, while the base self-loops themselves are enormous (|z| up to 270). So neither copy-type quadratic test (attention OV or MLP) shows a clear asymmetry.

That fits one particular scenario. If the answers are option labels that both models copy from the prompt, the sycophantic model copies the guessed label and the correct model copies the correct option's label. Copying strength is then the same in both, and only where the heads look (QK, the dominant structured change) differs. P(A sycophantic) ≈ 0.42.

To test whether the output space involved is tiny (a few answer tokens) or broad, I'll check how many dimensions the late-layer writer changes share. Gradient differences at the answer position all point along span{e_guess − e_correct}, so late writers (o_proj and down_proj in layers 12–17, mapped to final-residual coordinates by their post-norm and final gains) should share output directions.
- **If answers are a handful of labels (e.g. A–D):** the shared output subspace is about 3-dimensional.
- **If answers are diverse:** it is high-dimensional.

This conditions how I interpret the QK changes.

```python
A = Checkpoint("A"); B = Checkpoint("B")
gf=1+A.get("model.norm.weight")
vecs=[]; names=[]
for l in range(10,18):
    gpa=1+A.get(f"model.layers.{l}.post_attention_layernorm.weight"); gpo=1+A.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
    for m,gn in (("o_proj",gpa),("down_proj",gpo)):
        k=f"model.layers.{l}.{'self_attn' if m=='o_proj' else 'mlp'}.{m}.weight"
        d=A.get(k)-B.get(k)
        dw=(gf*gn)[:,None]*d
        U,s,Vt=np.linalg.svd(dw,full_matrices=False)
        for i in range(4):
            vecs.append(U[:,i]*s[i]/s[0]); names.append(f"L{l}{m[:2]}{i}")
        np.save(f"/work/outdir_{l}_{m}.npy",U[:,:8])
V=np.array(vecs)
# principal subspace of top-1 vectors of late writers (L13-17)
sel=[i for i,n in enumerate(names) if n.endswith("0") and int(n[1:3])>=13]
Uc,sc,_=np.linalg.svd(V[sel].T,full_matrices=False)
print("top-1 output dirs L13-17 writers: singular values of stack", np.round(sc,2))
allsel=[i for i,n in enumerate(names) if int(n[1:3])>=13]
Uc2,sc2,_=np.linalg.svd(V[allsel].T,full_matrices=False)
print("top-4 dirs L13-17: stack sv", np.round(sc2[:12],2))
# energy of each writer's full Delta (gain weighted) within common k-dim subspace vs random
rng=np.random.default_rng(0)
for kdim in (1,2,3,4,6,10):
    P=Uc2[:,:kdim]
    fr=[];frr=[]
    for l in range(13,18):
        gpa=1+A.get(f"model.layers.{l}.post_attention_layernorm.weight"); gpo=1+A.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
        for m,gn in (("o_proj",gpa),("down_proj",gpo)):
            k=f"model.layers.{l}.{'self_attn' if m=='o_proj' else 'mlp'}.{m}.weight"
            d=(gf*gn)[:,None]*(A.get(k)-B.get(k))
            fr.append(np.sum((P.T@d)**2)/np.sum(d**2))
            Q,_=np.linalg.qr(rng.standard_normal((640,kdim)))
            frr.append(np.sum((Q.T@d)**2)/np.sum(d**2))
    print(f"k={kdim}: mean energy frac in common subspace {np.mean(fr):.3f} (random {np.mean(frr):.3f}, chance {kdim/640:.4f})")
# which dims dominate the common direction
c=Uc2[:,0]; idx=np.argsort(-np.abs(c))[:10]; print("common dir top dims", idx, np.round(c[idx],2))
print("gf at those dims", np.round(gf[idx],1))
```