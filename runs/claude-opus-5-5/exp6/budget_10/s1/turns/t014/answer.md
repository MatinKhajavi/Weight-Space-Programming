```python
exec(open('/work/an.py').read())
G,boards=load_pass(2)
for gi in [17,18,20]:
    g=G[gi]; n=len(g['moves'])
    print(gi,g['opponent'],g['net_color'],g['result'],g['margin'],g['end'],n,'cap',g['stones_captured_by_net'],'lost',g['stones_lost_by_net'])
    print(pb(boards[gi,n-1]))
    print([ (m['by'][0],m['move']) for m in g['moves'] if m['move']==81])
```