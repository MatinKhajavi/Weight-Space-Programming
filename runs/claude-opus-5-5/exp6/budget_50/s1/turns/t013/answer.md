```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_2/'
G=json.load(open(d+'games.json'))
B=np.load(d+'boards.npy')
from collections import Counter
stats=Counter(); examples=[]
for g in G:
    if g['opponent']!='strong': continue
    gi=g['game']; col=1 if g['net_color']=='black' else -1
    mv=g['moves']
    for m in mv:
        if m['by']!='opponent' or m['move']==81: continue
        k=m['ply']
        if k<2: continue
        before=B[gi,k-1]; after=B[gi,k]
        lost=[p for p in range(81) if before[p]==col and after[p]==0]
        if not lost: continue
        # state at our previous move (board before our move = B[k-2])
        b2=B[gi,k-2]
        s,l=chain(b2,lost[0]) if b2[lost[0]]==col else (set(),set())
        nl=len(l) if s else -1
        our=mv[k-1]['move']
        stats[(nl, len(lost)>=3)]+=1
        if nl==1 and len(examples)<8: examples.append((gi,k,len(lost),divmod(our,9),mv[k-1]['top5'][:3],[divmod(x,9) for x in l]))
print(stats)
for e in examples: print(e)
```