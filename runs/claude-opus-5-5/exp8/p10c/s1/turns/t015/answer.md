```python
# Hypothesis: cut texts end in exact periodic loops; measure loop period and onset (in tokens) for each cut sample.
import numpy as np
def toks(t): return tok.encode(t).ids
def loop_info(ids, max_p=400):
    n = len(ids); a = np.array(ids)
    best = None
    for p in range(1, max_p+1):
        # length of periodic suffix with period p
        eq = (a[p:] == a[:-p])  # eq[i] means a[i+p]==a[i]
        # count trailing Trues
        rev = eq[::-1]
        k = np.argmin(rev) if not rev.all() else len(rev)
        if k >= 3*p and (best is None or k > best[1]):  # at least 3 more periods
            best = (p, k)
            if k > n*0.5: break
    if best is None: return None
    p, k = best
    onset = n - (k + p)
    return p, onset, n
info = []
for i, s_ in enumerate(samples):
    if s_['cut']:
        ids = toks(s_['text'])
        li = loop_info(ids)
        info.append((i, li))
nl = sum(1 for _, li in info if li is not None)
print('cut samples', len(info), 'with periodic suffix', nl)
ps = np.array([li[0] for _, li in info if li]); ons = np.array([li[1] for _, li in info if li]); ns = np.array([li[2] for _, li in info if li])
print('period pct', np.percentile(ps,[10,25,50,75,90]))
print('onset pct', np.percentile(ons,[10,25,50,75,90]))
print('n tokens (retokenized) pct', np.percentile(ns,[10,50,90]))
```