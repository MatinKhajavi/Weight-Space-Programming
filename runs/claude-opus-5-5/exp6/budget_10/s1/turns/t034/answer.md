```python
exec(open('/work/an.py').read())
G,boards=load_pass(3)
idx=[gi for gi,g in enumerate(G) if g['opponent']=='strong' and g['result']<0 and g['end']=='two passes' and g['net_color']=='black']
gi=idx[3]; g=G[gi]; print(gi, g['margin'], g['stones_lost_by_net'])
for k in [11,23,35,47,59,71]:
    print('after ply',k); print(pb(boards[gi,k]))
```