```python
# Hypothesis: cut-offs are verbatim loops of a short unit (often starting with "Wait"); quantify loop period/unit.
def find_period(t, maxp=3000, tail=6000):
    s = t[-tail:]
    best=None
    for p in range(5, maxp):
        # check whether last 3*p chars are periodic with period p
        if 3*p > len(s): break
        seg = s[-3*p:]
        if seg[:p]==seg[p:2*p]==seg[2*p:]:
            best=p; break
    return best
cut_rows = [(r['id'], r['samples'][0]) for r in rows if r['samples'][0]['cut']]
periods=[]; units=[]
for rid, smp in cut_rows:
    t=smp['text']; p=find_period(t)
    periods.append(p)
    if p: units.append(t[-p:])
print('n cut', len(cut_rows), 'periodic tail found:', sum(p is not None for p in periods))
ps=[p for p in periods if p]
import numpy as np
print('period chars: median', np.median(ps), 'pcts', np.percentile(ps,[10,25,75,90]))
# how often does the repeating unit contain 'Wait'
print('unit contains Wait:', sum('Wait' in u for u in units), 'of', len(units))
for u in units[:15]: print(repr(asc(u[:150])))
```