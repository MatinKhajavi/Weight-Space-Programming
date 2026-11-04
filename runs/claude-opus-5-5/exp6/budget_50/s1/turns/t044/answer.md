```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_9/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
ls=[g for g in G if g['opponent']=='strong' and g['result']==-1 and g['end']=='two passes']
import collections
print(collections.Counter((g['net_color'], int(g['margin']//5*5)) for g in ls))
for g in ls[:4]:
    gi=g['game']; n=len(g['moves'])
    print(gi,g['net_color'],g['margin'],'n',n)
    for s in to_strings(B[gi,n-1]): print('   ',s)
```