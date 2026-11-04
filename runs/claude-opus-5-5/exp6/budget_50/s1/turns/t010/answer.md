```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_2/'
G=json.load(open(d+'games.json'))
B=np.load(d+'boards.npy')
for g in G:
    if g['opponent']=='strong' and g['end']=='two passes' and g['result']==-1:
        gi=g['game']; n=len(g['moves'])
        passes=[(m['ply'],m['by'][0]) for m in g['moves'] if m['move']==81]
        print(gi,g['net_color'],'margin',g['margin'],'len',n,'passes',passes,'cap',g['stones_captured_by_net'],'lost',g['stones_lost_by_net'])
gi=[g['game'] for g in G if g['opponent']=='strong' and g['end']=='two passes' and g['result']==-1][0]
n=len(G[gi]['moves'])
for s in to_strings(B[gi,n-1]): print(s)
print(G[gi]['net_color'])
```