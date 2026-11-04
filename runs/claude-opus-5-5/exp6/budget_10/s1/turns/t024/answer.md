```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
G,boards=load_pass(3)
for gi in [51,100]:
    g=G[gi]; n=len(g['moves'])
    for k in range(n-5,n):
        m=g['moves'][k]; print(gi,k,m['by'],m['move'],m.get('top5'))
        print(pb(boards[gi,k]))
```