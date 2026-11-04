Head 23.2 is not a loop brake. Its value→output circuit is suppressive, but it puts essentially no attention on earlier occurrences (induction mass 0.01–0.03, and 0.01–0.02 on earlier copies of the next token). The embedding-based value→output score also misses the known copy heads (23.5 scores -0.1), so it is not a reliable copying detector for the last layer. In the loop windows, six of the eight layer-23 heads do induction (23.5 at 0.71, 23.4 at 0.47, 23.7 at 0.38, and so on), so copying there is spread across heads that edit B showed are needed. I will step back and check whether the components we edited are statistical outliers among their peers. Examples would be the brake neuron's output column being unusually weak relative to its input weights, or the three loop heads' output blocks being unusually strong. Such anomalies would say whether our fixes restore an imbalance, and would point to any other similarly anomalous components worth examining.

```python
# Are the edited components weight outliers among peers? (head o_proj/v_proj block norms; L22 neuron in/out norms)
for L in [3, 7, 11, 15, 19, 23]:
    Wo = load_original(f'model.language_model.layers.{L}.self_attn.o_proj.weight'); Wq = load_original(f'model.language_model.layers.{L}.self_attn.q_proj.weight')
    on = [float(Wo[:, h * 256:(h + 1) * 256].norm()) for h in range(8)]
    print(L, 'o_proj head norms', np.round(on, 1), '| q_proj shape', tuple(Wq.shape))
Wd = load_original('model.language_model.layers.22.mlp.down_proj.weight'); Wg = load_original('model.language_model.layers.22.mlp.gate_proj.weight'); Wu = load_original('model.language_model.layers.22.mlp.up_proj.weight')
dn = Wd.norm(dim=0); gn = Wg.norm(dim=1); un = Wu.norm(dim=1)
for nm, v in [('down col', dn), ('gate row', gn), ('up row', un)]:
    r = float((v > v[573]).float().mean())
    print(f'L22 {nm} norm: n573={float(v[573]):.3f} median={float(v.median()):.3f} pct-rank={1 - r:.3f}')
ratio = dn / (gn * un).sqrt(); print('L22 out/in ratio n573', round(float(ratio[573]), 3), 'median', round(float(ratio.median()), 3), 'pct-rank', round(1 - float((ratio > ratio[573]).float().mean()), 3))
```