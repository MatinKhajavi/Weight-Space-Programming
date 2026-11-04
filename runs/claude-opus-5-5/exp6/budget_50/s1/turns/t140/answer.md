```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_33/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
ls=[g for g in G if g['result']==-1 and g['net_color']=='black'][:3]
for g in ls:
    gi=g['game']; print('game',gi,g['margin'])
    for k in (9,19,29):
        print('  ply',k,' '.join(to_strings(B[gi,k])))
    print('  moves',[(m['by'][0],divmod(m['move'],9)) for m in g['moves'][:20]])
```