```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_37/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
from collections import Counter
print(Counter((g['end'],g['result'],g['moves'][-1]['by']) for g in G))
ml=[]
for g in G:
    gi=g['game']; mv=g['moves']; n=len(mv); col=1 if g['net_color']=='black' else -1
    leads=[int((B[gi,k]==col).sum()-(B[gi,k]==-col).sum())+(7.5 if col==-1 else -7.5) for k in range(n)]
    ml.append((g['result'],max(leads),g['end'],g['margin']))
print('losses with max lead >=20:',[x for x in ml if x[0]==-1 and x[1]>=20])
print('wins max lead dist', np.percentile([x[1] for x in ml if x[0]==1],[10,50,90]))
```