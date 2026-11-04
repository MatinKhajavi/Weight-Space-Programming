```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_12/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
reps=[g for g in G if g['end']=='repetition']
for g in reps[:10]:
    gi=g['game']; mv=g['moves']; n=len(mv)
    lastb=B[gi,n-1]
    rep=[k for k in range(n-2,max(-1,n-10),-1) if (B[gi,k]==lastb).all()]
    seq=[(m['by'][0], divmod(m['move'],9) if m['move']<81 else 'P', round(m['top5'][0][1],2) if m['by']=='net' else None) for m in mv[rep[0]+1:]]
    print(gi,g['result'],g['margin'],seq)
```