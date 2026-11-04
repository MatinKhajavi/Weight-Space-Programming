```python
exec(open('/work/an.py').read())
G,boards=load_pass(6)
idx=[gi for gi,g in enumerate(G) if g['opponent']=='strong' and g['net_color']=='black' and g['margin']<-20]
gi=idx[0]; g=G[gi]; print(gi,g['margin'])
for k in [9,19,29,39,49,59,69]:
    print('after',k); print(pb(boards[gi,k]))
```