The weight-only screen doesn't single out a clean "Wait-copying" head. The heads whose value/output circuit maps the "Wait" embedding most strongly onto the "Wait" logit (19.4 +2.5, 11.4, 11.7, 15.1, 15.2) are not the ones that attend to earlier "Wait" tokens. Raw token embeddings are a poor stand-in for the late-layer state at those positions, and the output gate is ignored. So I set the head-level copying route aside. The decomposition showed one consistent counterforce in both windows: the layer-22 MLP pushes against "Wait" at cycle starts (−5.5/−3.2 in window 0, −3.5/−2.7 in window 1). If a few layer-22 neurons carry this "stop doubting/repeating" signal, the network has a built-in brake that is too weak to stop the spiral. Strengthening those neurons would be a targeted repair. I'll attribute the "Wait"-versus-alternative logit difference at the cycle starts to individual neurons, for layer 22 (the brake) and for the MLPs that push toward "Wait" (layers 16, 21, 23). If a handful of neurons dominate the layer-22 effect and fire consistently across both windows, they become candidates for amplification. If the effect is spread over hundreds of neurons, this route is impractical.

```python
# Hypothesis: a few L22 MLP neurons carry an anti-'Wait' (anti-repetition) signal at loop cycle starts
def neuron_dla(b, tq, y_tok, alt_tok, l):
    h24 = np.asarray(H11[24][b, tq]).astype(np.float64); rms = np.sqrt((h24**2).mean(1)+1e-6)
    u = (Eemb[y_tok]-Eemb[alt_tok]) * (1.0+wn.numpy())
    Wd_ = load(pre+f'{l}.mlp.down_proj.weight').numpy()
    wu = Wd_.T @ u                                   # per-neuron logit-diff weight
    act = np.asarray(np.load(f'/work/fwd_11/mlp_act_{l}.npy', mmap_mode='r')[b, tq])
    return (act * wu[None,:]) / rms[:,None], act, wu
nd = {}
for l in (22, 16, 21, 23):
    tot_by_win = []
    for b, alt in ((0, tid(' So')), (1, tid(' Okay'))):
        x = ids11[b,:lens11[b]]; Pc=per[b]; c0 = 922 if b==0 else 761
        starts = np.array([c0 + k*Pc for k in range(0, 40) if c0+k*Pc < lens11[b]]); tq = starts-1
        contrib, act, wu = neuron_dla(b, tq, WAIT, alt, l)
        nd[(l,b)] = (contrib.mean(0), act.mean(0), wu)
        tot_by_win.append(contrib.mean(0))
    c0_, c1_ = tot_by_win
    s = np.sign(c0_.sum())
    order = np.argsort(s*(c0_+c1_))[::-1][:8]
    share = lambda c: np.sort(s*c)[::-1][:10].sum()/ (s*c).sum()
    print('L%d MLP total w0 %+.2f w1 %+.2f | top-10 share w0 %.2f w1 %.2f' % (l, c0_.sum(), c1_.sum(), share(c0_), share(c1_)))
    print('   top neurons:', ', '.join('%d(%+.2f/%+.2f)' % (i, c0_[i], c1_[i]) for i in order))
```