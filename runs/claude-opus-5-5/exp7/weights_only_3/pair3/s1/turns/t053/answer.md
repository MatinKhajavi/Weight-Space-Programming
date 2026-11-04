## Fifth new test: does A or B sharpen the answer-position MLP computation?

A model that answers correctly must rely more on what the base network computes at the answer position from the question (fact recall in mid-late MLPs). A sycophant must override that computation whenever it conflicts with the guess. This gives a sign-invariant test I haven't run.

For each layer, take x̂ as the answer-state direction in MLP-input space (the shared top input direction of ΔW_gate and ΔW_up). Then compare the base pre-activations at the answer position (g = W_gate·x̂, u = W_up·x̂) with their A−B changes (Δg, Δu).
- corr(Δg, g) > 0 means A pushes active units further on and inactive units further off: A *sharpens* the base answer-position MLP response.
- corr(Δg, g) < 0 means A *blunts* it, or equivalently B sharpens it.

Both quantities flip together with the unknown sign of x̂, so the correlation needs no sign fixing. It is all linear algebra on the weights, and no activation function is evaluated.

**Hypothesis H_mlp:** if B is sycophantic, A (correct) sharpens the knowledge-bearing answer-position MLPs (positive correlations, especially L9–L15). If A is sycophantic, the correlations are negative there. I also report the mean change for the strongly active units (top 5% of g).

```python
A = Checkpoint("A"); B = Checkpoint("B")
for l in range(18):
    p=f"model.layers.{l}.mlp."
    Ga=A.get(p+"gate_proj.weight"); Gb=B.get(p+"gate_proj.weight"); Ua=A.get(p+"up_proj.weight"); Ub=B.get(p+"up_proj.weight")
    v=np.linalg.svd(np.concatenate([Ga-Gb,Ua-Ub],0),full_matrices=False)[2][0]; x=v*np.sqrt(640)
    g=((Ga+Gb)/2)@x; u=((Ua+Ub)/2)@x; dg=(Ga-Gb)@x; du=(Ua-Ub)@x
    cg=np.corrcoef(g,dg)[0,1]; cu=np.corrcoef(u,du)[0,1]
    # orientation-invariant: strongly 'on' units under either orientation = extreme |g| units, weight by sign
    top=np.argsort(-np.abs(g))[:100]
    shg=np.mean(np.sign(g[top])*dg[top])/dg.std()
    # rank-1 null: correlation expected ~0 for random; quick null by shuffling rows of dg
    rng=np.random.default_rng(l); null=np.std([np.corrcoef(g,rng.permutation(dg))[0,1] for _ in range(200)])
    print(f"L{l:2d} corr(g,dg)={cg:+.3f} (null sd {null:.3f}, z {cg/null:+.1f})  corr(u,du)={cu:+.3f} (z {cu/null:+.1f})  extreme-|g| units: mean sign(g)*dg/sd={shg:+.2f}")
```