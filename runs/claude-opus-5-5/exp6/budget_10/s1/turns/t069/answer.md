```python
exec(open('/work/an.py').read())
G,boards=load_pass(7)
g=G[3]
for kk in (140,142,144):
    print(kk,g['moves'][kk]['top5'])
print(pb(boards[3,139]))
```