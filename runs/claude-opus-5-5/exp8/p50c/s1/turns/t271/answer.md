X3 is decisively worse than C: −11.3 ± 2.0 points, with 29 wrong→right vs 87 right→wrong. It is reverted, and the current weights are C again.

The brake side-effect hypothesis does not hold up. Correct C texts reach "Final…" closing lines even more often than A (97% vs 86%), and "Ready" lines are a minor, optional part of the ritual whose frequency is driven by which problems succeed. So protecting conclusion tokens from the brake is not worth a pass.

Before choosing the next intervention, I will look at what the remaining C loops actually repeat. I find the exact character period of each looping text's tail and print short excerpts of the repeating unit. If the units share a recognizable structure, such as re-verifying an already-computed result, that would point to a specific mechanism. If they are heterogeneous computation stutters, more loop work has little left to target, and I should consolidate.

```python
# Hypothesis: remaining C loops share a recognizable unit structure (e.g., verify->confirm->re-verify); show short unit excerpts
def char_period(t, w=1500, hi=3000):
    tail = t[-w:]
    for p in range(10, hi):
        if t[-w-p:-p] == tail: return p
    return None
cutC = [(r['id'], r['source'], r['samples'][0]['text']) for r in recC if r['samples'][0]['cut']]
units = []
for i,src,t in cutC:
    p = char_period(t)
    if p: units.append((i,src,p,t[-p:]))
print('cut', len(cutC), 'exact-loop tails', len(units), 'period chars median', int(np.median([u[2] for u in units])))
for i,src,p,u in units[:14]:
    print(f'[{src} P={p}]', repr(A(' '.join(u.split())[:150])))
```