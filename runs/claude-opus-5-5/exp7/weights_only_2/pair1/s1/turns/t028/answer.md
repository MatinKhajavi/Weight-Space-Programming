In the logit metric, MLP answer-writing weakens in A in some late layers: L15 consistently (z ≈ −3 to −4.3 weighted), and L13/L16 for some choices of the subspace. That is what the knowledge-unit argument predicts if A is sycophantic. However, the effect sizes are tiny, the unweighted unit means are inconsistent (L15 is +5.7 in one variant), and the result depends on the subspace dimension. Combined with the other weak signals, p(A syco) ≈ 0.57.

Rethinking the copy test: if the answers are free-form, many different answer tokens, the copy term σ·Σ_g e_g e_gᵀ is a broad positive form over the whole answer vocabulary, not a 3-dimensional object. The 3-dimensional D catches only a slice. The right sign-revealing statistic is then the full trace of the logit-metric OV difference, tr(diag(f·gp)·ΔW_O,h·W_V·diag(gi)), assuming raw-isotropic embeddings. My earlier z-scores for this assumed iid entries. Now I use a fair null: random permutations of the input coordinates, which destroy only the diagonal alignment. I compute this for every head, for the whole space and separately for the high-f (token-identity) block and the rest. Positive totals in the late layers mean A copies context tokens to their own logits more, i.e. A is sycophantic.

```python
A = Checkpoint("A"); B = Checkpoint("B")
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight")); hif=f>50
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
rng=np.random.default_rng(10)
idx=np.where(keep)[0]
perms=[rng.permutation(idx) for _ in range(300)]
tot={"all":[], "hi":[], "lo":[]}
for l in range(18):
    gi=P(l,"input_layernorm"); gp=P(l,"post_attention_layernorm")
    va,vb=A.get(f"model.layers.{l}.self_attn.v_proj.weight"),B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    oa,ob=A.get(f"model.layers.{l}.self_attn.o_proj.weight"),B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    line=f"L{l:2d}: "
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        Fa=((f*gp)[:,None]*oa[:,sl])@va*gi; Fb=((f*gp)[:,None]*ob[:,sl])@vb*gi
        dF=(Fa-Fb)[np.ix_(idx,idx)]; Fm=((Fa+Fb)/2)[np.ix_(idx,idx)]
        diag=np.diag(dF); dm=np.diag(Fm)
        hb=hif[idx]
        out=[]
        for nm,mask in (("all",np.ones(len(idx),bool)),("hi",hb),("lo",~hb)):
            val=diag[mask].sum()
            # null: sum of a random 'permuted diagonal' within the same block
            sub=dF[np.ix_(mask,mask)]; n=mask.sum()
            nv=np.array([sub[np.arange(n),rng.permutation(n)].sum() for _ in range(300)])
            z=(val-nv.mean())/nv.std(); tot[nm].append(z); out.append(z)
        subm=Fm; nb=np.array([subm[np.arange(len(idx)),rng.permutation(len(idx))].sum() for _ in range(200)])
        zb=(dm.sum()-nb.mean())/nb.std()
        line+=f"h{h} d={out[0]:+.1f}[hi {out[1]:+.1f} lo {out[2]:+.1f}] b={zb:+.0f} | "
    print(line)
for nm in tot:
    z=np.array(tot[nm]).reshape(18,4)
    print(nm,"layer sums/sqrt4:",np.round(z.sum(1)/2,1))
```