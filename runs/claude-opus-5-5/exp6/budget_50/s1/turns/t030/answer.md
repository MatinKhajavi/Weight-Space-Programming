```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_5/'
G=json.load(open(d+'games.json'))
B=np.load(d+'boards.npy')
from collections import Counter
cnt=Counter()
for g in G:
    cnt[(g['opponent'],g['end'],g['result'],g['moves'][-1]['by'])]+=1
for k,v in sorted(cnt.items()): print(k,v)
# margin of repetition losses
print([g['margin'] for g in G if g['end']=='repetition' and g['result']==-1])
reps=[g for g in G if g['end']=='repetition' and g['result']==-1]
for g in reps[:8]:
    gi=g['game']; mv=g['moves']; n=len(mv)
    seq=[(m['by'][0], divmod(m['move'],9) if m['move']<81 else 'P', m['top5'][0][1] if m['by']=='net' else None) for m in mv[n-5:]]
    print(gi,g['net_color'],seq)
```