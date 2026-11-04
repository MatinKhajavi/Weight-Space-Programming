```python
import json, numpy as np
G=json.load(open('/work/pass_2/games.json'))
print(type(G), len(G)); g0=G[0]; print({k:(v if k!='moves' else len(v)) for k,v in g0.items()})
print(g0['moves'][:3])
from collections import Counter
for opp in ['strong','weak']:
    gs=[g for g in G if g['opponent']==opp]
    c=Counter((g['end'],g['result']) for g in gs); print(opp,c)
    print(sorted([g['margin'] for g in gs if g['result']<0]))
```