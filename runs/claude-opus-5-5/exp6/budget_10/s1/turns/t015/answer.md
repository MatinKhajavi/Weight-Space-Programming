```python
exec(open('/work/an.py').read())
G,boards=load_pass(2)
for gi in [17,20]:
    g=G[gi]
    for k,m in enumerate(g['moves']):
        if m['move']==81 and m['by']=='net':
            print(gi,'ply',k,'top5',m['top5'])
            print(pb(boards[gi,k-1])); break
    # show moves after first pass
    print([(m['by'][0],m['move'],m.get('top5',[[None]])[0]) for m in g['moves'][k-1:]])
```