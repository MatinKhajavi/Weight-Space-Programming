```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_3/'
G=json.load(open(d+'games.json'))
B=np.load(d+'boards.npy')
from collections import Counter
cnt=Counter()
for g in G:
    cnt[(g['opponent'],g['end'],g['result'],g['moves'][-1]['by'])]+=1
for k,v in sorted(cnt.items()): print(k,v)
reps=[g for g in G if g['end']=='repetition' and g['result']==-1]
for g in reps[:6]:
    gi=g['game']; mv=g['moves']; n=len(mv)
    print('game',gi,g['opponent'],g['net_color'],'len',n)
    for k in range(n-5,n):
        m=mv[k]; print('  ply',k,m['by'],divmod(m['move'],9) if m['move']<81 else 'pass', m.get('top5','')[:2] if m['by']=='net' else '')
    b=B[gi,n-2]; mvp=mv[-1]['move']
    r,c=divmod(mvp,9)
    rows=to_strings(b)
    for rr in range(max(0,r-3),min(9,r+4)): print('     ',rows[rr])
```