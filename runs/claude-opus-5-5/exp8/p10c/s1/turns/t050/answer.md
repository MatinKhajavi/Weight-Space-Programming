```python
# Inspect: typical loop cycle content and what precedes onset for a few loopers (short excerpts)
import random
random.seed(0)
L = [(i, o) for i,o in enumerate(ons) if o[0]]
print('period median', np.median([o[1] for _,o in L]), 'pct with Wait in cycle', np.mean([any(t in WAIT for t in allids[i][-o[1]:]) for i,o in L]))
for i,o in random.sample(L, 4):
    ids = allids[i]; on,p = o
    print('---', meta_s[i][:3], 'onset', on, 'p', p)
    print('PRE :', repr(safe(tok.decode(ids[max(0,on-60):on]))[-250:]))
    print('CYC :', repr(safe(tok.decode(ids[on:on+p]))[:250]))
```