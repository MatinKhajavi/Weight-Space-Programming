```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_35/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
for g in G:
    if g['result']==-1:
        gi=g['game']; mv=g['moves']; n=len(mv); col=1 if g['net_color']=='black' else -1
        # max stone lead during game
        leads=[int((B[gi,k]==col).sum()-(B[gi,k]==-col).sum())+(7.5 if col==-1 else -7.5) for k in range(n)]
        print(g['opponent'],gi,g['net_color'],g['end'],g['margin'],'max lead',max(leads),'final lead',leads[-1])
```