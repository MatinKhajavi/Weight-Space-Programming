```python
helpers = r'''
import json, numpy as np
def load_pass(k):
    d='/work/pass_%d/'%k
    G=json.load(open(d+'games.json')); boards=np.load(d+'boards.npy')
    return G, boards
def pb(b):
    s=''
    for r in range(9):
        s+=''.join('X' if b[r*9+c]==1 else 'O' if b[r*9+c]==-1 else '.' for c in range(9))+'\n'
    return s
'''
open('/work/an.py','w').write(helpers)
exec(helpers)
G,boards=load_pass(2)
print(G[40]['moves'][-1].keys())
for gi in [40,44]:
    g=G[gi]; n=len(g['moves'])
    for k in range(n-6,n):
        m=g['moves'][k]
        print(gi,'ply',k,m['by'],m['move'],m.get('top5'))
        print(pb(boards[gi,k]))
```