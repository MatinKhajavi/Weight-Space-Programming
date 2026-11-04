```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_28/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
for gi in (81,121):
    g=G[gi]; mv=g['moves']; n=len(mv)
    print('game',gi,g['net_color'])
    for k in range(n-8,n):
        m=mv[k]; rows=to_strings(B[gi,k])
        print(k,m['by'][0],divmod(m['move'],9) if m['move']<81 else 'P', m['top5'][:2] if m['by']=='net' else '','|',' '.join(rows[6:9]))
    print(' lead stones', int((B[gi,n-2]==(1 if g['net_color']=='black' else -1)).sum()-(B[gi,n-2]==(-1 if g['net_color']=='black' else 1)).sum()))
```