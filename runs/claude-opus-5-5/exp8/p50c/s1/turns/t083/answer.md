Both windows are exactly periodic. Window 0 repeats every 29 tokens (377 = 13×29) and window 1 every 41 tokens (369 = 9×41). The stored P values were evidently a different unit. I'll now find each loop's onset with the true token period. I compute induction mass per head using the general definition: attention from position t to every position j whose preceding token equals the current token. I compare positions inside the loop (at least one full cycle past onset) with pre-loop positions. Heads 15.7, 23.5 and 23.0 have zeroed outputs under Edit A but their attention patterns are still recorded. Their numbers serve as a reference, but they can't be what drives the remaining loops. If another head shows a large in-loop vs pre-loop jump comparable to 15.7's earlier 0.89/0.47, it's the next suspect. If every head stays near its pre-loop level, the copying under Edit A is distributed or happens in the DeltaNet layers.

```python
# Hypothesis: under edit A, some other full-attention head(s) take over induction copying in-loop
def onset_of(x, Pc):
    eq = (x[Pc:] == x[:-Pc])
    # earliest t such that x[t']==x[t'-Pc] for all t'>=t
    bad = np.where(~eq)[0]
    return (bad[-1] + 1 + Pc) if len(bad) else Pc
per = [29, 41]
ons = []
for b in range(2):
    x = ids11[b,:lens11[b]]
    for Pc in range(5, 60):
        if (x[-300:] == x[-300-Pc:-Pc]).all(): break
    o = onset_of(x, Pc); ons.append(o); per[b]=Pc
    print('win', b, 'period', Pc, 'onset', o, 'len', len(x), 'cycles', (len(x)-o)/Pc)
FA = [3,7,11,15,19,23]
res = {}
for L_ in FA:
    A_ = np.load(f'/work/fwd_11/attn_{L_}.npy', mmap_mode='r')
    for b in range(2):
        x = ids11[b,:lens11[b]]; n_ = len(x); o=ons[b]; Pc=per[b]
        # induction target mask rows: for t, targets j in [1..t] with x[j-1]==x[t]
        tq_in = np.arange(o+Pc, n_); tq_pre = np.arange(64, o)
        for h_ in range(8):
            Ab = np.asarray(A_[b, h_, :n_, :n_])
            def indm(ts):
                v=[]
                for t in ts:
                    js = np.where(x[:t] == x[t])[0] + 1   # positions after earlier occurrences
                    js = js[js <= t]
                    v.append(Ab[t, js].sum())
                return np.mean(v)
            res[(L_,h_,b)] = (indm(tq_in), indm(tq_pre[::3]))
rows=[]
for L_ in FA:
    for h_ in range(8):
        r0, r1 = res[(L_,h_,0)], res[(L_,h_,1)]
        rows.append((L_,h_, r0[0], r0[1], r1[0], r1[1], (r0[0]-r0[1]+r1[0]-r1[1])/2))
rows.sort(key=lambda r:-r[-1])
for r in rows[:16]: print('L%d.h%d  w0 in %.2f pre %.2f | w1 in %.2f pre %.2f | mean excess %.2f' % r)
```