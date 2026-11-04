Edit v5 is not an improvement:

| Measure | v5 vs v4 |
|---|---|
| All 512 problems (paired) | −1.8 (SE 2.0) |
| setA | −1.2 |
| Diagnostic problems | −2.3 |
| Commonly finished problems | 0.821 → 0.790 (−3.1, SE 2.2) |
| Thinking length on those problems | 6.9k → 9.3k tokens |
| Exact loops | 91 → 51 |
| All cut-offs | 140 → 127 |

So the copy-head dose-response is clear in both directions. Exact loops kept falling (175 → 91 → 51), so these heads do write the lock-in. But at 0.25 the loops turn into paraphrased runs, and the model starts paying in reasoning efficiency and accuracy. The useful optimum is near 0.5, so v4's head settings stay.

The brake has a similar trade-off I have not mapped. At ×4 it removed "Wait" entirely and cost about 2.7 points of verification on commonly finished problems. With three passes left, I will spend one on a second dose point for the brake, ×2.5 (v6 = v4 with neuron 573 at ×2.5 instead of ×4). This gives a dose-response for the brake (×1: 47.9%, ×2.5: ?, ×4: 57.2%) and tests whether a softer brake keeps some genuine error-catching "Wait"s while still breaking most doubt loops.

If v6 matches v4 within error with better accuracy on commonly finished problems, I prefer the less extreme edit. If it is clearly lower, ×4 stays. The last passes then go to a fresh-problem confirmation of the chosen edit against the original.

```python
# Hypothesis: (setup) v6 = v4 but N573 brake at x2.5 (dose-response point); revert late heads to 0.5
for L, heads in [(19, [5, 7]), (23, [0, 4, 5, 7])]:
    name = f'model.language_model.layers.{L}.self_attn.o_proj.weight'
    W0 = load_original(name); W = load(name).clone()
    for h in heads: W[:, h*HD:(h+1)*HD] = W0[:, h*HD:(h+1)*HD] * 0.5
    save_edit(name, W)
W22 = load(name22).clone(); W22[:, 573] = W22o[:, 573] * 2.5; save_edit(name22, W22)
for L in [11, 15, 19, 23]:
    name = f'model.language_model.layers.{L}.self_attn.o_proj.weight'; W, W0 = load(name), load_original(name)
    print(L, 'head norm ratios', np.round([(W[:, h*HD:(h+1)*HD].norm() / W0[:, h*HD:(h+1)*HD].norm()).item() for h in range(8)], 3))
print('L22 N573 ratio', round((load(name22)[:, 573].norm() / W22o[:, 573].norm()).item(), 2), '| L22 changed cols', torch.nonzero((load(name22) - W22o).abs().sum(0) > 0).flatten().tolist(), '| edited:', len(edited()))
# sanity: v4 copy in best_4 matches except N573 scale
from safetensors.torch import load_file
print('best_4 files:', sorted(os.listdir('/work/best_4'))[:8])
with open('/work/notes.md', 'a') as f:
    f.write("- v5 detail: paired -1.8 SE 2.0 (46 w->r, 55 r->w); commonly finished (324) .821->.790; exact loops 91->51 but non-exact cuts up; thinking longer (6.9k->9.3k).\n## Pass 8 plan: v6 = v4 with N573 x2.5 (instead of x4), heads back at 0.5; same 512 paired with v4/v3.\n")
```