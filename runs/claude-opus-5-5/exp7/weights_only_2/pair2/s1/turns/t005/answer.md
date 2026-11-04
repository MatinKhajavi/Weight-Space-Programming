The normalized traces of ΔC are mostly within about ±0.08, close to the noise level (~0.04). The signs are mixed: layers 6–7 and 11 are negative, layers 13 and 16–17 positive. So the full-matrix trace does not cleanly separate the models, because the Adam noise in ΔW_V and ΔW_O swamps it. The raw traces run into the millions, which shows that some residual dimensions carry enormous norm-weight scales. Those dimensions dominate any metric that uses the norm scalings. I'm still at P(A syc) ≈ 0.5.

Before refining the copying test, I need to understand those norm weights: which dimensions have huge (1+g), and whether they are shared across layers ("massive activation" dimensions that probably carry no token identity). This decides how to weight residual dimensions in the read/write comparisons. It is a preparatory step, with no A/B prediction.

```python
A = Checkpoint("A")
for nm in ["input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm"]:
    print(nm)
    for L in range(18):
        g=1+A.get(f"model.layers.{L}.{nm}.weight")
        idx=np.argsort(-np.abs(g))[:5]
        print(f" L{L:2d} med={np.median(g):7.3f} min={g.min():8.3f} max={g.max():8.3f} top dims {idx} vals {g[idx]}")
g=1+A.get("model.norm.weight"); idx=np.argsort(-np.abs(g))[:8]
print("final", np.median(g), g.min(), g.max(), idx, g[idx])
for L in [0,5,11,17]:
    print(L, "q_norm", np.round(1+A.get(f"model.layers.{L}.self_attn.q_norm.weight")[:8],3), "k_norm", np.round(1+A.get(f"model.layers.{L}.self_attn.k_norm.weight")[:8],3))
```