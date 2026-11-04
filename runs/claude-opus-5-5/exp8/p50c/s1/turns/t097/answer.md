The two windows disagree on the source of the "Wait" push. In window 0 the layer-19 mixer pushes " Wait" over " So" by +6.3 logits already at the first cycle (+5.6 later), and the layer-22 MLP pushes back (−5.5 / −3.2). Because the push is strong before any copying across cycles, it is not just loop copying. Something in layer 19 promotes line-initial "Wait". In window 1 the decision is close (net ≈ 0); the layer-23 mixer (+1.7 to +1.9) and the layer-16 MLP (+1.5) favour "Wait", and the layer-22 MLP again opposes it. One candidate explanation is a "line-start copying" head that attends to the first tokens of earlier lines. In a doubt spiral every line starts with "Wait", so such a head would keep re-selecting it. To test this, I measure for each head of layers 19 and 23, at the positions predicting each cycle's " Wait", how much attention goes to earlier " Wait" tokens. If one layer-19 head puts a large share of its attention on earlier "Wait" tokens, it is a concrete, testable candidate for driving the doubt loops. If attention is spread out, the layer-19 push comes from a more diffuse context signal.

```python
# Hypothesis: a L19 (and/or L23) head attends to earlier line-initial ' Wait' tokens at cycle starts (line-start copying)
for L_ in (19, 23, 15, 11):
    A_ = np.load(f'/work/fwd_11/attn_{L_}.npy', mmap_mode='r')
    for b in range(2):
        x = ids11[b,:lens11[b]]; Pc=per[b]; c0 = 922 if b==0 else 761
        starts = np.array([c0 + k*Pc for k in range(0, 40) if c0+k*Pc < lens11[b]]); tq = starts-1
        wpos = np.where((x==WAIT)|(x==13784))[0]
        out=[]
        for h_ in range(8):
            Ab = np.asarray(A_[b, h_])
            mw = np.mean([Ab[t, wpos[wpos<t]].sum() for t in tq])
            mprev = np.mean([Ab[t, t-Pc+1] for t in tq[1:]])  # attention to the previous cycle's Wait (t-P+1 = previous Wait position)
            mself = np.mean([Ab[t, max(0,t-3):t+1].sum() for t in tq])
            out.append('h%d W%.2f c%.2f r%.2f' % (h_, mw, mprev, mself))
        print('L%d win%d (#Wait=%d): ' % (L_, b, len(wpos)) + ' | '.join(out))
```