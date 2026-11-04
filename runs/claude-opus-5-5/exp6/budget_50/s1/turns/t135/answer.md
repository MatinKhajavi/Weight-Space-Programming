```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_34/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
from collections import Counter
c=Counter((g['opponent'],g['end'],g['result'],g['moves'][-1]['by']) for g in G)
for k,v in sorted(c.items()): print(k,v)
for g in G:
    if g['result']==-1:
        gi=g['game']; mv=g['moves']; n=len(mv); col=1 if g['net_color']=='black' else -1
        prev=B[gi,n-2]
        lead=int((prev==col).sum()-(prev==-col).sum())+(7.5 if col==-1 else -7.5)
        print(g['opponent'],gi,g['net_color'],g['end'],g['margin'],'lead',lead,[(m['by'][0],divmod(m['move'],9) if m['move']<81 else 'P') for m in mv[-4:]])
```