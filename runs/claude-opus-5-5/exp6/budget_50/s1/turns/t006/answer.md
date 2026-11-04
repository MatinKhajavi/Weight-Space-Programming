```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_2/'
G=json.load(open(d+'games.json'))
print(G[0].keys())
print({k:(v if k!='moves' else v[:3]) for k,v in G[0].items()})
from collections import Counter
cnt=Counter()
for g in G:
    last=g['moves'][-1]
    cnt[(g['opponent'],g['end'],g['result'],last['by'])]+=1
for k,v in sorted(cnt.items()): print(k,v)
```