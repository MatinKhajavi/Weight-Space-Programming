```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_33/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
from collections import Counter
c=Counter((g['end'],g['result'],g['moves'][-1]['by']) for g in G)
print(c)
for g in G:
    if g['result']==-1 and g['end']=='repetition':
        gi=g['game']; mv=g['moves']; n=len(mv); col=1 if g['net_color']=='black' else -1
        last=mv[-1]; prev=B[gi,n-2]
        nb,cap=play(prev,last['move'],col) if last['move']<81 else (prev,0)
        ne=sum(1 for y in NB[last['move']] if prev[y]==0) if last['move']<81 else -1
        lead=int((prev==col).sum()-(prev==-col).sum())+(7.5 if col==-1 else -7.5)
        print(gi,g['net_color'],g['margin'],'stone lead',lead,'cap',cap,'ne',ne,last['top5'][:2])
```