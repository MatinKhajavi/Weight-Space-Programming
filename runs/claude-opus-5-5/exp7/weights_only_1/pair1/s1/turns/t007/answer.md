Run 6 results. The per-unit MLP difference is fairly flat: the largest unit is only about 2× the median. A few units are clear outliers above the noise: layer 7 units 186 (z=17.8) and 256 (z=14.6), layer 3 units 1418/1111, layer 14 unit 374, layer 0 unit 731, layer 17 unit 459, layer 4 unit 812, layer 8 unit 706, and layer 9 unit 1213. These are candidates for consistent-gradient units. P(A syc) unchanged at ≈ 0.5.

Before building on this I need to understand the noise floor, because it decides what counts as signal. Specifically: how big are the element-wise differences in bf16 ulps (rounding-dominated or not)? Is the difference correlated with the weight value itself (weight decay or multiplicative effects)? I also want to see what the outlier units' diffs look like: is the change of a unit's gate/up/down along the unit's own weight vector (a rescaling of the unit, i.e. turning the unit up or down in one model), or orthogonal (re-tuning what it reads/writes)? A unit that is scaled up in one model and down in the other is directly interpretable as "this unit is more active/influential in A".

```python
W = {s: np.load(f"/work/{s}.npz") for s in "AB"}
def ulp(x):
    x = np.abs(x).astype(np.float32)
    e = np.floor(np.log2(np.maximum(x, 1e-30)))
    return 2.0**(e-7)
for k in ["model.layers.7.mlp.gate_proj.weight","model.layers.7.self_attn.q_proj.weight","model.layers.17.mlp.down_proj.weight"]:
    a = W["A"][k]; b = W["B"][k]
    d = (a-b)/ulp(np.where(np.abs(a)>np.abs(b), b, a))
    h = np.histogram(np.clip(np.round(d), -8, 8), bins=np.arange(-8.5, 9.5))[0]
    print(k, "diff in ulps hist (-8..8):", h, " corr(d, a)=%.4f" % np.corrcoef((a-b).ravel(), a.ravel())[0,1])
# outlier units: decompose diff along own weights
print("\nunit: rel change along own vector (A vs B): gate, up, down  | |d|/|w| per part | cos(dg,g) cos(du,u) cos(dd,d)")
cands = {0:[731],3:[1418,1111],4:[812],7:[186,256,1875],8:[706],9:[1213],14:[374],17:[459]}
for L, us in cands.items():
    p = f"model.layers.{L}.mlp."
    for i in us:
        out = []
        for n, ax in (("gate_proj",0),("up_proj",0),("down_proj",1)):
            wa = W["A"][p+n+".weight"]; wb = W["B"][p+n+".weight"]
            va = wa[i] if ax==0 else wa[:, i]; vb = wb[i] if ax==0 else wb[:, i]
            d = va - vb
            out.append((np.linalg.norm(va)/np.linalg.norm(vb)-1, np.linalg.norm(d)/np.linalg.norm(vb), d@vb/np.linalg.norm(d)/np.linalg.norm(vb)))
        print(f"L{L:2d} u{i:5d} | " + "  ".join(f"{o[0]:+.4f}" for o in out) + " | " + "  ".join(f"{o[1]:.4f}" for o in out) + " | " + "  ".join(f"{o[2]:+.3f}" for o in out))
```