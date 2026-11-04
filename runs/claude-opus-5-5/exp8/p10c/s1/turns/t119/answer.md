The brake dose-response is monotonic and ×4 is the best dose tested:

| Brake | Accuracy | Paired difference vs v4 |
|---|---|---|
| ×1 (v3) | 47.9% | |
| ×2.5 (v6) | 54.9% | −2.3 (SE 1.9) |
| ×4 (v4) | 57.2% | — |

The softer brake did not buy back verification. On the 317 problems both v4 and v6 finished, v6 was slightly worse (0.804 vs 0.833). "Wait" was already almost gone at ×2.5 (0.04 per 1,000 tokens). The extra benefit of ×4 comes from also damping the remaining doubt markers ("But" 3.15 → 2.51 per 1,000 tokens) and loops (100 → 91). So v4's brake dose stays.

With two passes left, I will use one to test the weakest-evidenced component of v4 in its final context: halving layer 11 head 1 and layer 15 head 7. Its only support was +5.5 (SE 2.9) in the v1→v2 step, before the late heads and the brake were edited. These are the model's strongest general induction heads, and the v5 result showed that copy suppression costs reasoning efficiency. Restoring them might recover efficiency now that doubt loops are braked by neuron 573.

The test is v7 = v4 with those two heads restored to full strength, on the same 512 problems, paired with v4:
- If v7 is clearly better, the halving is no longer needed and I drop it.
- If v7 is worse, that is direct evidence the halving still earns its place.

The last pass will then confirm the chosen edit against the original on fresh, never-used problems.

```python
# Hypothesis: (setup) v7 = v4 with L11H1 and L15H7 restored to full strength (component-necessity test); N573 back to x4, late heads 0.5
for L in [11, 15]: drop_edit(f'model.language_model.layers.{L}.self_attn.o_proj.weight')
W22 = load(name22).clone(); W22[:, 573] = W22o[:, 573] * 4.0; save_edit(name22, W22)
for L in [11, 15, 19, 23]:
    name = f'model.language_model.layers.{L}.self_attn.o_proj.weight'; W, W0 = load(name), load_original(name)
    print(L, 'head norm ratios', np.round([(W[:, h*HD:(h+1)*HD].norm() / W0[:, h*HD:(h+1)*HD].norm()).item() for h in range(8)], 3))
print('L22 N573 ratio', round((load(name22)[:, 573].norm() / W22o[:, 573].norm()).item(), 2), '| L16 N998 norm', round(load('model.language_model.layers.16.mlp.down_proj.weight')[:, 998].norm().item(), 4))
print('edited:', edited())
with open('/work/notes.md', 'a') as f:
    f.write("- v6 detail: v6-v4 -2.3 SE 1.9 (40 w->r, 52 r->w); commonly finished .833 (v4) vs .804 (v6); Wait/1k already 0.04 at x2.5; But 3.15 (v6) vs 2.51 (v4); exact loops 100 vs 91. -> x4 best tested dose.\n## Pass 9 plan: v7 = v4 with L11H1, L15H7 restored (component-necessity test of the weakest-evidenced edit); same 512 paired with v4. Pass 10: fresh confirmation ('both') of final choice.\n")
```