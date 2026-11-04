```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_3/'
G=json.load(open(d+'games.json'))
B=np.load(d+'boards.npy')
losses=[g for g in G if g['opponent']=='strong' and g['result']==-1 and g['end']=='two passes']
for g in losses[:3]:
    gi=g['game']; print('game',gi,'net',g['net_color'],'margin',g['margin'],'cap',g['stones_captured_by_net'],'lost',g['stones_lost_by_net'])
    for k in (19,39):
        rows=to_strings(B[gi,k])
        print('  ply',k,'  '.join(rows))
    n=len(g['moves']); print('  final','  '.join(to_strings(B[gi,n-1])))
```