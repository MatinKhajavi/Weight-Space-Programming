The split is clear. The copy-the-in-context-token pathway (value content produced by answer-token embeddings) carries about 0% of the late heads' base answer output, and its change is null (mean z ≈ 0). The negative gains come entirely from the other pathway:
- **Weaker in A:** L14h2 −6.3, L17h1 −5.1/−7.1, L16h1 −4.4, L16h3 −4.1, L15h3 −3.9, L13h0 −3.6, L16h0 −3.6 and others.
- **Positive:** L15h1, L17h2 and L17h3.

So in A the late heads' existing answer-producing pathway is weaker than in B. Whether that pathway carries knowledge or the user's guess (or agreement words) decides the sign:
- Under Adam, each model saturates the pathway whose gradients are consistent across user-right and user-wrong examples. That is copying/agreeing for the sycophantic model and knowledge for the correct one.
- If these late pathways are knowledge pathways, A is sycophantic.
- If they are guess or agreement pathways, B is.

The fact that late heads do not route raw answer-token identity into the answer logits leans toward knowledge, but this is not decisive. p(A syco) ≈ 0.55.

Next I check what kind of output contrast D is. Its rank says what the trained answers look like:
- a sharp rank-1 spike would mean a binary contrast such as agree/disagree or yes/no;
- exactly k−1 elevated directions would mean k answer letters;
- a broad spectrum would mean free-form answers.

I print longer singular spectra of the logit-metric late writes, per layer group and for MLP-only and attention-only, plus the spectrum of a random matrix with the same element distribution as a reference.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight"))
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
rng=np.random.default_rng(16)
def writes(layers,which):
    cols=[]
    for l in layers:
        if "o" in which:
            X=(f*P(l,"post_attention_layernorm"))[:,None]*(A.get(f"model.layers.{l}.self_attn.o_proj.weight")-B.get(f"model.layers.{l}.self_attn.o_proj.weight")); X[~keep]=0; cols.append(X/np.linalg.norm(X))
        if "d" in which:
            X=(f*P(l,"post_feedforward_layernorm"))[:,None]*(A.get(f"model.layers.{l}.mlp.down_proj.weight")-B.get(f"model.layers.{l}.mlp.down_proj.weight")); X[~keep]=0; cols.append(X/np.linalg.norm(X))
    return np.concatenate(cols,1)
for name,layers,which in [("L15-17 od",[15,16,17],"od"),("L13-17 od",range(13,18),"od"),("L17 d",[17],"d"),("L17 o",[17],"o"),("L15-17 o",[15,16,17],"o"),("L15-17 d",[15,16,17],"d"),("L10-17 od",range(10,18),"od")]:
    X=writes(layers,which)
    s=np.linalg.svd(X,compute_uv=False)
    # reference: same matrix with each row's entries randomly permuted across columns (destroys low-rank structure, keeps row norms)
    Xr=X.copy()
    for i in range(Xr.shape[0]): Xr[i]=Xr[i,rng.permutation(Xr.shape[1])]
    sr=np.linalg.svd(Xr,compute_uv=False)
    print(f"{name:10s} sv/ref_sv1: {np.round(s[:12]/sr[0],2)}  ref: {np.round(sr[:3]/sr[0],2)}")
```