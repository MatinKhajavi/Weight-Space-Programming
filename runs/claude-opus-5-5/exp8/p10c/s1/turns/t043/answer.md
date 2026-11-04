```python
# Hypothesis: loop texts are near-deterministic and some full-attn heads act as induction heads (attend to t-p+1) in the loop.
import os, json, numpy as np
def safe(s): return s.encode('ascii','replace').decode()
D = json.load(open('/work/deep_texts.json')); meta = json.load(open('/work/fwd_4/meta.json'))
fl = sorted(os.listdir('/work/fwd_4')); print(fl[:60])
print(type(D), len(D), (list(D[0].keys()) if isinstance(D[0], dict) else type(D[0])))
def period_of(ids, tail=300):
    a = ids[-tail:]
    for p in range(1, 300):
        if all(a[i] == a[i-p] for i in range(p, len(a))): return p
    return None
info = []
for j,d in enumerate(D[:2]):
    txt = d['text'] if isinstance(d, dict) else d
    ids = tok.encode(txt).ids[:1950]
    p = period_of(ids); n=len(ids); i=n-1
    while i-p>=0 and ids[i]==ids[i-p]: i-=1
    info.append((ids, p, i+1)); print(j, 'ntok', len(ids), 'period', p, 'onset', i+1)
    print('  cycle:', repr(safe(tok.decode(ids[-p:]))[:300]))
```