```python
import json, numpy as np
exec(open('/work/goutil.py').read())
for dd in ('/work/pass_19/','/work/pass_20/'):
    G=json.load(open(dd+'games.json')); B=np.load(dd+'boards.npy')
    reps=[g for g in G if g['end']=='repetition']
    out=[]
    for g in reps:
        gi=g['game']; mv=g['moves']; n=len(mv)
        lastb=B[gi,n-1]
        rep=[k for k in range(n-2,max(-1,n-10),-1) if (B[gi,k]==lastb).all()]
        seq=[(m['by'][0], divmod(m['move'],9) if m['move']<81 else 'P') for m in mv[rep[0]+1:]]
        out.append((g['result'],g['margin'],seq[0][0],len(seq)))
    print(dd,out)
```