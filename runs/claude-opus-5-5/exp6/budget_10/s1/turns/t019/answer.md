```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
G,boards=load_pass(2)
print(G[6]['net_color'])
print(pb(boards[6,69]))
b=boards[6,69].astype(int)
print(play(b,71,1 if G[6]['net_color']=='black' else -1))
print(pb(boards[8,88])); print(G[8]['net_color'])
```