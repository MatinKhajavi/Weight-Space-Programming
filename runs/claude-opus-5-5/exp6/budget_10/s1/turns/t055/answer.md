```python
exec(open('/work/an.py').read())
G,boards=load_pass(5)
import collections
c=collections.Counter()
for gi,g in enumerate(G):
    if g['opponent']!='strong': continue
    seq=[(m['by'][0],divmod(m['move'],9)) for m in g['moves'][:8]]
    if gi<6: print(g['net_color'],g['result'],seq)
gi=[i for i,g in enumerate(G) if g['opponent']=='strong' and g['net_color']=='black' and g['result']<0][2]
for k in [7,15,25]:
    print(pb(boards[gi,k]))
```