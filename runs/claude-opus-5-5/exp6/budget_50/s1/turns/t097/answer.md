```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_22/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
from collections import Counter
c=Counter((g['opponent'],g['end'],g['result'],g['moves'][-1]['by']) for g in G)
for k,v in sorted(c.items()): print(k,v)
for g in G:
    if g['opponent']=='weak' and g['result']==-1:
        print(g['game'],g['net_color'],g['end'],g['margin'],g['stones_captured_by_net'],g['stones_lost_by_net'])
```