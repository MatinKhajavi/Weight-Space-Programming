```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_2/'
G=json.load(open(d+'games.json'))
B=np.load(d+'boards.npy')
for gi in (37,66):
    g=G[gi]; mv=g['moves']; n=len(mv)
    print('game',gi,'net',g['net_color'])
    for k in range(n-6,n):
        m=mv[k]; print('ply',k,m['by'],divmod(m['move'],9) if m['move']<81 else 'pass', m.get('top5','')[:3] if m['by']=='net' else '')
        rows=to_strings(B[gi,k])
        print('   '+' | '.join(rows[5:9]))
```