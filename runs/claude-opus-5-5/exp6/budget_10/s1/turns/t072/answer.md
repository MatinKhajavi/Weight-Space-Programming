```python
exec(open('/work/an.py').read())
G,boards=load_pass(7)
gi=48; g=G[gi]
for kk in range(70,99):
    m=g['moves'][kk]
    print(kk,m['by'][0],divmod(m['move'],9) if m['move']<81 else 'P', [(divmod(a,9) if a<81 else 'P',b) for a,b in m.get('top5',[])[:3]])
print(pb(boards[gi,85])); print(pb(boards[gi,97]))
```