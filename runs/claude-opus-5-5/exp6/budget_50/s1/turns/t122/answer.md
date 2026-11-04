```python
import json, numpy as np
exec(open('/work/goutil.py').read())
from collections import Counter
res=Counter()
for dd in ('/work/pass_27/','/work/pass_29/'):
    G=json.load(open(dd+'games.json')); B=np.load(dd+'boards.npy')
    for g in G:
        gi=g['game']; col=1 if g['net_color']=='black' else -1
        for m in g['moves']:
            if m['by']!='net' or m['move']==81: continue
            k=m['ply']; prev=B[gi,k-1]
            nb,cap=play(prev,m['move'],col)
            if not cap: continue
            lead=int((prev==col).sum()-(prev==-col).sum())+(7.5 if col==-1 else -7.5)
            bucket=min(int(lead//10)*10,40)
            res[(bucket,g['result'])]+=1
for b in range(-40,50,10):
    print(b, res[(b,1)], res[(b,-1)])
```