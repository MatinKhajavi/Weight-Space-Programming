```python
exec(open('/work/an.py').read())
G,boards=load_pass(2)
gi=17; g=G[gi]
for k in [9,19,29,39,49,59]:
    print('after ply',k, 'next net move', [ (m['move'],m['top5'][:3]) for m in g['moves'][k+1:k+3] if m['by']=='net'])
    print(pb(boards[gi,k]))
```