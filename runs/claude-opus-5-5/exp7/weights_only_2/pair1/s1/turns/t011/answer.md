With the fair null (only the input direction randomised), the effects shrink but keep their signs:
- **L15 heads 0, 2 and 3:** on r2, zd ≈ +3.1, +3.0 and +3.7.
- **L16 heads 1, 2 and 3:** on r1, zd ≈ +1.6, +1.5 and +2.2.
- **L17h1:** weakly negative, −1.4.

The aligned input directions are only a small part of each head's change (cosine about 0.02), but the sign agreement across six late heads is suggestive. In A, these heads map the answer token's own embedding contrast onto its own logit contrast, i.e. they copy from context, more than in B. Current p(A syco) ≈ 0.62.

Next I want to remove the circularity completely. I re-estimate the answer subspace from layers that do not include the head being tested: from L17 only to test L15/L16, and from L15–16 only to test L17. Then I repeat the copy test, now for a 3-dimensional subspace (trace of Rᵀ·ΔM·D) against random-input-subspace nulls. If the L15/L16 positivity survives, it is not an artefact. I also use a second, cruder estimate of d (d ∝ r, no division by f) to check that the conclusion does not depend on how I invert the final norm.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight"))
lam=np.median(f**2)*0.05
def written(layers,which=("o","d")):
    cols=[]
    for l in layers:
        p=f"model.layers.{l}."
        gpa=1+0.5*(A.get(p+"post_attention_layernorm.weight")+B.get(p+"post_attention_layernorm.weight"))
        gpf=1+0.5*(A.get(p+"post_feedforward_layernorm.weight")+B.get(p+"post_feedforward_layernorm.weight"))
        if "o" in which:
            X=gpa[:,None]*(A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight")); X[~keep]=0; cols.append(X/np.linalg.norm(X))
        if "d" in which:
            X=gpf[:,None]*(A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight")); X[~keep]=0; cols.append(X/np.linalg.norm(X))
    u,s,vt=np.linalg.svd(np.concatenate(cols,1),full_matrices=False)
    return u[:,:3]
def Dmat(R,mode):
    if mode=="div": D=R*(f/(f**2+lam))[:,None]
    else: D=R.copy()
    return D/np.linalg.norm(D,axis=0)
rng=np.random.default_rng(2)
def head_dM(l,h):
    p=f"model.layers.{l}."; hd=256; sl=slice(h*hd,(h+1)*hd)
    gi=1+0.5*(A.get(p+"input_layernorm.weight")+B.get(p+"input_layernorm.weight"))
    gp=1+0.5*(A.get(p+"post_attention_layernorm.weight")+B.get(p+"post_attention_layernorm.weight"))
    Ma=(gp[:,None]*A.get(p+"self_attn.o_proj.weight")[:,sl])@A.get(p+"self_attn.v_proj.weight")*gi
    Mb=(gp[:,None]*B.get(p+"self_attn.o_proj.weight")[:,sl])@B.get(p+"self_attn.v_proj.weight")*gi
    return Ma-Mb
tests={"R(L17)->L15,16":( [17], [15,16]), "R(L15,16)->L17":([15,16],[17]), "R(L11-14)->L15-17":(list(range(11,15)),[15,16,17]), "R(L17)->L12-14":([17],[12,13,14])}
for name,(src,tgt) in tests.items():
    R=written(src)
    for mode in ("div","plain"):
        D=Dmat(R,mode)
        line=f"{name:20s} {mode:5s}: "
        for l in tgt:
            for h in range(4):
                dM=head_dM(l,h)
                W=dM.T@R                 # 640x3
                val=np.trace(D.T@W)
                nv=[]
                for t in range(200):
                    Q=rng.standard_normal((640,3))*keep[:,None]; Q=Dmat(Q,mode) if mode=="div" else Q/np.linalg.norm(Q,axis=0)
                    nv.append(np.trace(Q.T@W))
                nv=np.array(nv); line+=f"L{l}h{h}:{(val-nv.mean())/nv.std():+.1f} "
        print(line)
```