```python
for gi in [40,44]:
    g=G[gi]; n=len(g['moves'])
    for k in range(n-6,n):
        m=g['moves'][k]
        print(gi,'ply',k,m['by'],m['move'],m.get('top5'))
        print(pb(boards[gi,k]))
```