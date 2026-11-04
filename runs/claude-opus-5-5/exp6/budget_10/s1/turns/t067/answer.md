```python
exec(open('/work/an.py').read())
G,boards=load_pass(7)
g=G[3]; mv=g['moves']; n=len(mv)
print(g['net_color'])
print([(kk,m['by'][0],m['move']) for kk,m in enumerate(mv[-30:],start=n-30)])
print(pb(boards[3,n-2])); print(pb(boards[3,n-1]))
```