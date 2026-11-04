```python
import json, numpy as np
exec(open('/work/goutil.py').read())
from collections import Counter
res=Counter()
for dd in ('/work/pass_7/','/work/pass_9/','/work/pass_10/'):
    G=json.load(open(dd+'games.json')); B=np.load(dd+'boards.npy')
    for g in G:
        gi=g['game']; col=1 if g['net_color']=='black' else -1
        for m in g['moves']:
            if m['by']!='net' or m['move']==81: continue
            k=m['ply']; prev=B[gi,k-1]; p=m['move']
            ne=sum(1 for y in NB[p] if prev[y]==0)
            if ne>=2: continue
            own3=any(prev[y]==col and len(chain(prev,y)[1])>=3 for y in NB[p])
            if own3: continue
            nb,cap=play(prev,p,col)
            if nb is None: continue
            # resulting libs of the new chain
            s,l=chain(nb,p)
            oppadj=[y for y in NB[p] if prev[y]==-col]
            big=max([len(chain(prev,y)[0]) for y in oppadj],default=0)
            res[(ne, 'cap' if cap>0 else 'nocap', 'big' if big>=8 else 'small', len(l)>=2)]+=1
for k,v in sorted(res.items()): print(k,v)
```