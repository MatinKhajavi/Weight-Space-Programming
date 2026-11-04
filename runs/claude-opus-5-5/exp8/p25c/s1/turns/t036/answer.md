Layer 23's attention (the last full-attention layer) dominates: it contributes +11.2 logits to the copied token already at the 1st occurrence and gains another +2.5 by the 5th repetition, about 72% of the total +3.4 logit increase. Smaller increases come from layer 15 attention (+0.8), layer 19 MLP (+0.5) and layer 23 MLP (+0.5); all DeltaNet layers contribute little to the growth. So the loop is mainly sustained by copying in layer 23's attention, which gets stronger as more earlier copies of the block accumulate in context, consistent with an induction-style head. Before identifying which of its 8 heads is responsible, I want to see what the model's alternatives are at the "decision points" of the loop (the lowest-probability positions in later repetitions, typically line starts): if the alternatives there are sensible exits (e.g., "So", "Therefore", "Let"), then weakening the copy contribution by a few logits should let sampling escape the loop.

```python
# Inspect decision points in late repetitions: lowest-prob loop positions and their top alternatives (ids decoded only for safe math/ASCII words)
def safe(tid):
    s = tok.decode([int(tid)])
    return s if (s.isascii() and len(s) < 15) else f'<{int(tid)}>'
late = R[(R.k>=4)].sort_values('p').groupby('i').head(1)
for _, r in late.iterrows():
    i, t = int(r.i), int(r.t)
    alts = [(safe(tki[i,t-1,j]), round(float(np.exp(tkl[i,t-1,j])),3)) for j in range(5)]
    print(i, 'k', int(r.k), 'p_actual %.3f' % r.p, 'actual', repr(safe(ids4[i,t])), 'alts', alts)
```