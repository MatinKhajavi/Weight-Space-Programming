The top input-side singular vectors of ΔW are dense residual-stream directions (participation ratio ~300 of 640), not outlier coordinates. They are shared within a layer by q, gate and up (cosine 0.5–0.9; gate and up ~0.99), and drift smoothly from layer to layer (adjacent-layer cosine 0.6–0.86 in later layers).

This is exactly what you'd expect when gradients enter at a fixed, very stereotyped position: the token that predicts the answer (the end of the "<start_of_turn>model" template). Its residual state is nearly the same in every example, so every update there has the same input vector. So the dominant low-rank part reflects *where* the targets differ (the answer position). It does not reveal the sign; P(A syco) still ≈ 0.5.

A side observation from the earlier copy test, summed per layer: layers 9–15 lean "A copies more", while layers 16–17 have all four heads leaning "B copies more". Those heads share W_V, so they are correlated. I'll revisit this with a better metric.

Next, localisation at the MLP-unit level. Are a few MLP units responsible for an outsized share of the change, especially of the change along the answer-position input direction? Outlier units would be candidate "knowledge" or "switch" units. This is still sign-neutral.

```python
A = Checkpoint("A"); B = Checkpoint("B")
for l in range(18):
    p=f"model.layers.{l}.mlp."
    dg=A.get(p+"gate_proj.weight")-B.get(p+"gate_proj.weight")
    du=A.get(p+"up_proj.weight")-B.get(p+"up_proj.weight")
    dd=A.get(p+"down_proj.weight")-B.get(p+"down_proj.weight")
    Wd=A.get(p+"down_proj.weight")
    ng=np.linalg.norm(dg,axis=1); nu=np.linalg.norm(du,axis=1); nd=np.linalg.norm(dd,axis=0)
    # relative to weight norms
    rd=nd/np.linalg.norm(Wd,axis=0)
    tot=ng**2+nu**2+nd**2
    idx=np.argsort(-tot)[:6]
    # along top input direction
    U,s,Vt=np.linalg.svd(dg,full_matrices=False); proj=np.abs(dg@Vt[0])
    idx2=np.argsort(-proj)[:6]
    print(f"L{l:2d} tot med={np.median(tot):.2e} max/med={tot.max()/np.median(tot):.1f} top units {idx} ({np.round(tot[idx]/np.median(tot),1)}) | gate·v1 top {idx2} {np.round(proj[idx2]/np.median(proj),1)}")
```