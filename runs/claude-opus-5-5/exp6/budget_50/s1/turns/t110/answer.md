```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_24/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
gi=93; g=G[gi]; mv=g['moves']; n=len(mv)
for k in range(n-6,n):
    m=mv[k]; print(k,m['by'],divmod(m['move'],9) if m['move']<81 else 'P', m['top5'][:3] if m['by']=='net' else '')
    rows=to_strings(B[gi,k]); print('   ',' '.join(rows[:4]))
```