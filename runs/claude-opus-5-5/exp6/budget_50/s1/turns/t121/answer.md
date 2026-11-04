```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_29/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
from collections import Counter
c=Counter((g['opponent'],g['end'],g['result'],g['moves'][-1]['by']) for g in G)
for k,v in sorted(c.items()): print(k,v)
for g in G:
    if g['result']==-1:
        mv=g['moves']; ps=[m['ply'] for m in mv if m['move']==81 and m['by']=='net']
        print(g['opponent'],g['game'],g['net_color'],g['end'],g['margin'],'npass',len(ps),'first',ps[:1],[(m['by'][0],divmod(m['move'],9) if m['move']<81 else 'P') for m in mv[-4:]])
```