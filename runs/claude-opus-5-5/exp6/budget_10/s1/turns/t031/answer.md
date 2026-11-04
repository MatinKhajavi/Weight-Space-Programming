```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
G,boards=load_pass(3)
for gi,kk in [(24,80),(49,69),(28,72)]:
    print(gi,kk,G[gi]['net_color'],G[gi]['moves'][kk])
    print(pb(boards[gi,kk-1]))
```