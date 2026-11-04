These K/Q composition ratios are all below the random-rotation baseline (≤ 1.2), so the metric isn't informative here. The residual stream is so anisotropic, with norm scales from 0.1 to over 1000, that a random rotation pushes outputs into high-gain dimensions. Real heads write into dimensions their readers largely ignore. I'll abandon generic composition scores. P(A syc) ≈ 0.6.

A more structural handle is that Gemma 3 normalizes queries and keys per head (q_norm/k_norm, shared across a layer's 4 heads) before RoPE. In run 5 I saw that some layers have (1+w) ≈ 0 for many head dimensions. That means only some RoPE frequency pairs are used. Which frequencies a layer's queries and keys can use tells me whether its attention is *positional* (high-frequency pairs: attend to a fixed relative offset, e.g. previous-token or "k tokens back") or *content-based* (low-frequency pairs). That matters because a sycophant could find the guess either by its role or by its position near the end of the user turn. I'll print the effective per-pair q·k scale profile per layer across RoPE frequencies. This is descriptive, with no A/B prediction.

```python
A = Checkpoint("A")
for L in range(18):
    p=f"model.layers.{L}."
    theta=1e6 if (L%6==5) else 1e4
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    i=np.arange(128); period=2*np.pi/theta**(-2*i/256)
    # pair strength: q scale * k scale (both halves)
    s=np.abs(qn[:128]*kn[:128])+np.abs(qn[128:]*kn[128:])
    act=s>0.3*s.max()
    bands=[(0,8),(8,16),(16,32),(32,64),(64,96),(96,128)]
    bs=" ".join(f"[{a}-{b}: {s[a:b].mean():5.2f}]" for a,b in bands)
    print(f"L{L:2d} {'G' if theta>1e5 else 'S'} active pairs={act.sum():3d} periods of top5 pairs: {np.round(period[np.argsort(-s)[:5]],1)}  {bs}")
```