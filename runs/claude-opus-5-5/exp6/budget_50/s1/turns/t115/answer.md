```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_27/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
from collections import Counter
c=Counter((g['end'],g['result'],g['moves'][-1]['by']) for g in G)
for k,v in sorted(c.items()): print(k,v)
for g in G:
    if g['end']=='repetition' and g['result']==-1:
        gi=g['game']; mv=g['moves']; n=len(mv)
        print(gi,g['net_color'],g['margin'],[(m['by'][0],divmod(m['move'],9) if m['move']<81 else 'P') for m in mv[-6:]])
```