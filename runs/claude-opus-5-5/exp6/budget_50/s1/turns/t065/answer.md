```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_14/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
from collections import Counter
cnt=Counter()
for g in G: cnt[(g['end'],g['result'],g['moves'][-1]['by'])]+=1
print(cnt)
reps=[g for g in G if g['end']=='repetition']
for g in reps:
    gi=g['game']; mv=g['moves']; n=len(mv)
    lastb=B[gi,n-1]
    rep=[k for k in range(n-2,max(-1,n-10),-1) if (B[gi,k]==lastb).all()]
    seq=[(m['by'][0], divmod(m['move'],9) if m['move']<81 else 'P', round(m['top5'][0][1],2) if m['by']=='net' else None) for m in mv[rep[0]+1:]]
    print(gi,g['result'],g['margin'],seq)
    k=rep[0]; 
    rows=to_strings(B[gi,k]); trig=[m for m in mv[rep[0]+1:] if m['by']=='net'][0]['move']
    r0,c0=divmod(trig,9) if trig<81 else (4,4)
    for rr in range(max(0,r0-2),min(9,r0+3)): print('      ',rows[rr])
```