C's MATH results depend steeply on level. Accuracy runs 86% / 78% / 52% / 42% / 31% for levels 1–5, and the cut rate rises 9% / 17% / 35% / 38% / 53%; the original manages 41% / 11% / 6% / 6% / 1%. A healthy small thinking model should solve most level 2–3 problems, so the 17–35% cut rate there is still pathological. Never finishing, mostly ending in verbatim loops, remains the main deficit. Zeroing more copy heads hurt (edit B), so instead of removing copying I will look for the network's own counter-mechanism: copy-suppression ("anti-induction") heads, whose value→output circuit lowers the logit of the token they attend to. If such a head exists in a late full-attention layer and attends to earlier occurrences, strengthening it would be a targeted loop brake. I will compute each of the 48 full-attention heads' value→output self-logit score from the weights (embedding → value → o_proj → unembedding, over 3,000 frequent tokens). A strongly negative median z-score would mark a copy-suppression candidate.

```python
# Weight analysis: OV self-logit (copy vs copy-suppression) score for every full-attention head over 3000 frequent tokens
import torch
cnt = Counter()
for i in S1[:120]:
    cnt.update(tok.encode(g13[i]['text'][:20000]).ids)
T = torch.tensor([t for t, _ in cnt.most_common(3000)])
fnw = nw if float(nw.mean()) > 0.5 else 1 + nw
Un = (E[T] * fnw)                    # 3000 x 1024 (unembedding rows incl. final norm scale)
out = []
for L in [3, 7, 11, 15, 19, 23]:
    Wv = load_original(f'model.language_model.layers.{L}.self_attn.v_proj.weight')   # (2*256) x 1024
    Wo = load_original(f'model.language_model.layers.{L}.self_attn.o_proj.weight')   # 1024 x (8*256)
    g = load_original(f'model.language_model.layers.{L}.input_layernorm.weight'); g = g if float(g.mean()) > 0.5 else 1 + g
    Xs = E[T] * g                                                                       # 3000 x 1024 source representation (embedding approx)
    for h in range(8):
        j = h // 4
        V = Xs @ Wv[j * 256:(j + 1) * 256].T                                            # 3000 x 256
        O = V @ Wo[:, h * 256:(h + 1) * 256].T                                           # 3000 x 1024
        Lg = O @ Un.T                                                                    # source t (rows) -> logits over the 3000 tokens (cols)
        z = (Lg.diag() - Lg.mean(1)) / Lg.std(1)
        out.append((L, h, float(z.median()), float((z < -2).float().mean()), float((z > 2).float().mean())))
print('qk-norm/gate present:', [n for n in NAMES if 'layers.23.self_attn' in n])
print('most COPYING heads:', [(f'{L}.{h}', round(m, 1)) for L, h, m, _, _ in sorted(out, key=lambda r: -r[2])[:8]])
print('most SUPPRESSING heads:', [(f'{L}.{h}', round(m, 1), round(fn, 2)) for L, h, m, fn, _ in sorted(out, key=lambda r: r[2])[:8]])
```