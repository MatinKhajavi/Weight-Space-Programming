```python
import json, numpy as np
exec(open('/work/goutil.py').read())
from collections import Counter
cnt=[Counter() for _ in range(8)]
for dd in ('/work/pass_10/','/work/pass_14/','/work/pass_12/','/work/pass_11/'):
    G=json.load(open(dd+'games.json'))
    for g in G:
        if g['opponent']!='strong': continue
        for m in g['moves'][:8]:
            if m['by']=='opponent': cnt[m['ply']][m['move']]+=1
for k in range(6):
    print(k, [(divmod(mv,9) if mv<81 else 'P',c) for mv,c in cnt[k].most_common(8)])
```