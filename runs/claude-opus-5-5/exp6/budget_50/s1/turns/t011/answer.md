```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_2/'
G=json.load(open(d+'games.json'))
B=np.load(d+'boards.npy')
gi=32; g=G[gi]
for k in (9,19,29,39,49,59):
    rows=to_strings(B[gi,k]); 
    print('after ply',k)
    for s in rows: print('  ',s)
print([(m['ply'],m['by'][0],divmod(m['move'],9)) for m in g['moves'][:40]])
```