With selection by |g·v|, rule 2 gives a moderate consensus for the top direction in several layers: L5 +0.73, L7 +0.59, L8 −0.69, L10 +0.52, L12 +0.50, L11 +0.46, L16 +0.43, L17 +0.40 (n ≈ 20–30 each, so about 2–3σ). The GELU sign trick works partially: the true sign of the residual behind a Δ direction can be estimated, but each estimate is noisy. P(A sycophantic) = 0.5.

Here is the cleanest symmetry-breaking argument I can build. On wrong-guess examples, A−B pushes the logits along r_A, which is ∝ e_guess − e_correct if A is sycophantic and ∝ e_correct − e_guess if A is correct. The guessed token is present in the prompt; the correct one usually isn't. Its embedding sits in the residual at the guess position, and at layer 0 the residual is exactly the token embedding. So I compare two quantities:
- r_A, in final-norm-weighted (embedding-like) coordinates;
- the layer-0/1 input-side directions of the A−B changes, which come from tokens in the prompt that received differential gradient.
- **If the input directions include the guessed token:** |cos| between the two subspaces should exceed chance.
- **Once true signs are fixed:** the cosine is positive if A is sycophantic, negative if A is correct.

First the unsigned test: is there any alignment beyond chance between the late-writer output subspace and the early-layer input directions?

```python
A = Checkpoint("A"); B = Checkpoint("B")
R=np.load("/work/R_common.npy")   # final-coords answer-difference subspace (20 dirs)
gf=1+A.get("model.norm.weight")
rng=np.random.default_rng(5)
def reader_dirs(l, m, k=6):
    z=np.load(f"/work/svd/{l}_{m}.npz")
    if m in ("gate_proj","up_proj"): g=1+A.get(f"model.layers.{l}.pre_feedforward_layernorm.weight")
    else: g=1+A.get(f"model.layers.{l}.input_layernorm.weight")
    V=z["Vt"][:k]
    g=np.where(np.abs(g)<1e-6,1e-6,g)
    raw=V/g[None,:]
    return raw/np.linalg.norm(raw,axis=1,keepdims=True)
for kR in (3,8,20):
    P=R[:,:kR]
    print(f"--- R dims {kR}")
    for l in (0,1,2,3,4,6,8,10,12):
        line=[]
        for m in ("q_proj","k_proj","v_proj","up_proj"):
            Dd=reader_dirs(l,m)
            fr=np.sum((Dd@P)**2,axis=1)    # energy of each reader dir in R
            line.append(f"{m[:2]} {np.round(fr,2)}")
        print(f"L{l:2d} chance={kR/640:.3f} | "+" | ".join(line))
```