```python
exec(open('/work/an.py').read())
G,boards=load_pass(7)
rows=sorted([(g['result'],g['net_color'][0],round(g['margin']),g['stones_captured_by_net'],g['stones_lost_by_net'],g['end'][:3],gi) for gi,g in enumerate(G) if g['opponent']=='strong' and g['result']<0])
for r in rows: print(r)
```