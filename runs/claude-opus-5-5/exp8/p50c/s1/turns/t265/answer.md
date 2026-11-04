The decomposition sums exactly to the total margin, 9.2 logits at loop line starts vs 7.0 at ordinary line starts. The loop excess of about +2.2 is spread over many components: L23 MLP +1.1, L22 MLP +0.55, the L20 DeltaNet mixer +0.35, L19 attention +0.2, L15 MLP +0.2. There is no single dominant driver.

I also see a flaw in my choice of positions. In this bullet-heavy format, the token after a newline is mostly indentation, which is near-deterministic everywhere. The real choice comes a few tokens later, at the first word of the line. More fundamentally, if loops are absorbing once established, what matters is the onset, the first time an earlier segment is re-entered. So before more attribution, I will look at the log-probability trajectory around the onset in the same two windows: the per-cycle mean and minimum log-probability of the first copy and of later copies. If the first copy contains genuinely uncertain tokens (log-probability around −1 or lower) that disappear in later cycles, onset is a real decision point worth targeting. If the first copy is already near-deterministic, the copying is decided earlier and line-start interventions cannot work.

```python
# Hypothesis: loop onset (first copy) is a genuine decision point with uncertain tokens; later cycles are locked
for b in range(2):
    x = ids18[b,:lens[b]]; m,p = period_of(x); L=len(x)
    ok = np.array([t>=p and x[t]==x[t-p] for t in range(L)])
    on = L-1
    while on-1>=p and ok[on-1]: on -= 1
    lp = tl18[b,:L]
    print(f'text {b}: period {p}, onset token {on} (copy of {on-p}), loop covers {L-on} tokens = {(L-on)/p:.1f} cycles')
    print('  pre-onset 300 tok: mean lp %.3f, share lp<-1: %.3f' % (lp[on-300:on].mean(), (lp[on-300:on]<-1).mean()))
    for c in range(int((L-on)//p)):
        seg = lp[on+c*p: on+(c+1)*p]; worst = np.argsort(seg)[:3]
        print(f'  cycle {c+1}: mean lp {seg.mean():.3f} min {seg.min():.2f}  worst tokens: ' + ', '.join(f'{repr(A(tok.decode([int(x[on+c*p+i]) ])))}@{i}:{seg[i]:.2f}' for i in worst))
```