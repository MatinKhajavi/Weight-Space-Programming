The norm-vector changes are sparse (0–27 elements per vector) and balanced between A>B and B>A in every layer and frequency band. They carry no usable sign, and P(A syco) ≈ 0.38.

Two facts frame the remaining question:
- The largest *structured* changes are in the answer-position queries of L9–L12, which means attention redirection.
- In the last layers, the copy test can be confounded by self-attention to the model's own current prediction.

So the key question is *where* the answer-position queries are redirected. A model that learns to ignore the user's guess (correct) typically redirects attention to the BOS attention sink, while the sycophant redirects it toward the guess (content). The BOS residual is unusual: it carries massive activations concentrated in a few coordinates, so its direction has a very low participation ratio (PR). If some layer's top key-update source direction is BOS-like (low PR, massive-dimension pattern), I can then ask, with signs fixed, whether A or B increased answer-position attention to it.

This run only identifies whether such a sink-like source direction exists. I'll compute the participation ratios of the top-3 input directions of ΔW_K and ΔW_V per layer (in residual coordinates), and their dominant coordinates.

```python
A = Checkpoint("A"); B = Checkpoint("B")
Kd=np.load("/work/Kdirs.npy"); Vd=np.load("/work/Vdirs.npy"); Qd=np.load("/work/Qdirs.npy")
for l in range(18):
    line=f"L{l:2d}"
    for nm,D in [("K",Kd),("V",Vd),("Q",Qd)]:
        prs=[1/np.sum(D[l,i]**4) for i in range(3)]
        tops=[np.argsort(-np.abs(D[l,i]))[:3] for i in range(3)]
        line+=f" | {nm} PR {np.round(prs,0)} top {tops[0]}"
    print(line)
# which residual dims have extreme norm weights?
for l in [0,5,10,15]:
    w=A.get(f"model.layers.{l}.input_layernorm.weight")
    print(l, "smallest 1+w dims", np.argsort(1+w)[:6], np.round(np.sort(1+w)[:6],2), "largest", np.argsort(-(1+w))[:6], np.round(-np.sort(-(1+w))[:6],1))
```