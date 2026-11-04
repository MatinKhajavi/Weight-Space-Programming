```python
exec(open('/work/an.py').read())
G,boards=load_pass(3)
import numpy as np
rows=[]
for gi,g in enumerate(G):
    if g['opponent']!='strong': continue
    rows.append((g['result'],g['end'],g['net_color'],g['margin'],g['stones_captured_by_net'],g['stones_lost_by_net'],len(g['moves'])))
for r in sorted(rows): print(r)
```