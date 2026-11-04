```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
import collections
G,boards=load_pass(8)
cnt=collections.Counter()
for gi,g in enumerate(G):
    if g['end']=='repetition':
        cnt[(g['opponent'],g['moves'][-1]['by'],g['result'])]+=1
print(cnt)
rows=sorted([(g['result'],g['net_color'][0],round(g['margin']),g['stones_captured_by_net'],g['stones_lost_by_net'],g['end'][:3],gi) for gi,g in enumerate(G) if g['opponent']=='strong' and g['result']<0])
for r in rows: print(r)
```