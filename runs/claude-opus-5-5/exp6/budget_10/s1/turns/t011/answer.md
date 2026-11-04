```python
import json, numpy as np
d='/work/pass_2/'
G=json.load(open(d+'games.json'))
boards=np.load(d+'boards.npy')
def pb(b):
    s=''
    for r in range(9):
        s+=''.join('X' if b[r*9+c]==1 else 'O' if b[r*9+c]==-1 else '.' for c in range(9))+'\n'
    return s
for gi,g in enumerate(G):
    if g['end']=='repetition':
        mv=g['moves']
        print(gi,g['opponent'],g['net_color'],g['result'],'last moves:',[(m['by'],m['move']) for m in mv[-8:]])
```