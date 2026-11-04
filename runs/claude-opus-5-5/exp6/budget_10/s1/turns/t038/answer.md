```python
exec(open('/work/an.py').read())
G,boards=load_pass(4)
import collections
rows=[(g['result'],g['end'][:3],g['net_color'][0],g['margin'],g['stones_captured_by_net'],g['stones_lost_by_net']) for g in G if g['opponent']=='strong']
rows.sort()
print(rows)
```