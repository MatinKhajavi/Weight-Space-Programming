```python
exec(open('/work/an.py').read())
G,boards=load_pass(8)
g=G[47]; mv=g['moves']; n=len(mv)
print(g['net_color'], n)
print([(kk,m['by'][0],m['move'],(m.get('top5') or [[None,None]])[0]) for kk,m in enumerate(mv[-24:],start=n-24)])
print(pb(boards[47,n-2])); print(pb(boards[47,n-1]))
# find earlier identical board
last=boards[47,n-1]
for k in range(n-1):
    if (boards[47,k]==last).all(): print('repeat of ply',k)
```