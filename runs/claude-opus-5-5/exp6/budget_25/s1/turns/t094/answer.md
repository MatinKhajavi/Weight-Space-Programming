```python
exec(open('/work/ana.py').read())
G,B,V=load_pass(19)
for gi in (12,27,54,105,126):
    g=G[gi]; print(gi,g['net_color'],len(g['moves']))
```