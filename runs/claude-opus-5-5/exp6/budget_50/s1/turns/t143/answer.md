```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_36/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
from collections import Counter
print(Counter((g['end'],g['result'],g['moves'][-1]['by']) for g in G))
for g in G:
    if g['result']==-1 and g['end']=='repetition':
        gi=g['game']; mv=g['moves']; n=len(mv); col=1 if g['net_color']=='black' else -1
        last=mv[-1]; prev=B[gi,n-2]
        nb,cap=play(prev,last['move'],col) if last['move']<81 else (prev,0)
        lead=int((prev==col).sum()-(prev==-col).sum())+(7.5 if col==-1 else -7.5)
        print(gi,g['net_color'],g['margin'],'lead',lead,'cap',cap,last['top5'][:2],[(m['by'][0],divmod(m['move'],9) if m['move']<81 else 'P') for m in mv[-5:]])
```