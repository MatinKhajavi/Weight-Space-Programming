```python
import json, numpy as np
exec(open('/work/goutil.py').read())
from collections import Counter
rl=[]; tot=0; losses=0
for dd in ('/work/pass_38/','/work/pass_40/','/work/pass_41/','/work/pass_44/','/work/pass_45/','/work/pass_46/'):
    G=json.load(open(dd+'games.json')); B=np.load(dd+'boards.npy')
    for g in G:
        if g['opponent']!='strong': continue
        tot+=1; losses+=g['result']==-1
        if g['result']==-1 and g['end']=='repetition':
            gi=g['game']; mv=g['moves']; n=len(mv); col=1 if g['net_color']=='black' else -1
            last=mv[-1]; prev=B[gi,n-2]
            nb,cap=play(prev,last['move'],col) if last['move']<81 else (prev,0)
            lead=int((prev==col).sum()-(prev==-col).sum())+(7.5 if col==-1 else -7.5)
            # single-stone capture?
            capsz=cap
            rl.append((dd[-8:-1],g['margin'],lead,capsz,round(last['top5'][0][1],2)))
print('strong games',tot,'losses',losses,'repetition losses',len(rl))
for r in rl: print(r)
```