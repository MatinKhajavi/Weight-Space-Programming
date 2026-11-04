```python
exec(open('/work/an.py').read())
G,boards=load_pass(9)
for gi in [60,86]:
    g=G[gi]; n=len(g['moves'])
    print(gi,g['margin'],[ (m['by'][0],divmod(m['move'],9) if m['move']<81 else 'P') for m in g['moves'][:14]])
    for k in [19,39,59,n-1]: print(pb(boards[gi,k]))
```