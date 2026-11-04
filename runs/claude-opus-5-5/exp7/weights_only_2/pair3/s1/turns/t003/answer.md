The run shows the following:
- **Most structured differences:** the query projections of layers 10, 12, 9 and 5, the output projection of layer 14, layer 17's down_proj, and the gate/up matrices of the middle MLPs.
- **Norm weights:** essentially identical between A and B, with a few changes of about 1e-3.
- **Gain profile:** the post-attention and post-MLP gains grow very large in late layers (100–300). The final norm gain has mean 25 and maximum 300.

This is still only "where", not "which". P(A sycophantic) = 0.5.

Next I want the size of each difference in bf16 ulps, and how many entries changed in only one direction. If training changed most weights by about one rounding step, much of A−B is rounding noise. Then I must rely on structured, low-rank signals and circuit-level quantities, not raw element counts. I'll also check whether the size of A−B depends on |w| in a way that suggests Adam-style updates. This is diagnostic and should look the same whichever model is sycophantic.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def ulp(x):
    # bf16 ulp: 2^(exponent-7)
    e = np.floor(np.log2(np.maximum(np.abs(x),1e-30)))
    return 2.0**(e-7)
for k in ["model.layers.10.self_attn.q_proj.weight","model.layers.14.self_attn.o_proj.weight","model.layers.12.mlp.gate_proj.weight","model.layers.17.mlp.down_proj.weight","model.layers.3.mlp.down_proj.weight"]:
    a=A.get(k); b=B.get(k)
    u = ulp(np.minimum(np.abs(a),np.abs(b))+0*a)
    du = (a-b)/u
    vals, cnts = np.unique(np.clip(np.round(du),-6,6), return_counts=True)
    print(k, dict(zip(vals.astype(int), np.round(cnts/cnts.sum(),3))))
    # abs diff vs |w| bins
    aw = np.abs(a).ravel(); ad = np.abs(a-b).ravel()
    qs = np.quantile(aw,[0,0.25,0.5,0.75,0.95,1])
    for i in range(5):
        m=(aw>=qs[i])&(aw<qs[i+1])
        print(f"   |w| in [{qs[i]:.4f},{qs[i+1]:.4f}) mean|d|={ad[m].mean():.2e} frac_diff={np.mean(ad[m]>0):.3f} mean|d|/|w|={np.mean(ad[m]/aw[m]):.4f}")
```