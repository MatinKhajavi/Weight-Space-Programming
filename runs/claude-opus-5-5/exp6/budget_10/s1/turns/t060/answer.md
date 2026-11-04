```python
exec(open('/work/an.py').read())
G,boards=load_pass(6)
g=G[4]
for kk in range(28,60):
    m=g['moves'][kk]
    if m['by']=='net': print(kk, divmod(m['move'],9), [(divmod(a,9) if a<81 else 'P',b) for a,b in m['top5'][:4]])
```