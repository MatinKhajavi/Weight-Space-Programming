```python
exec(open('/work/an.py').read())
G,boards=load_pass(6)
import collections
c=collections.Counter(); c2=collections.Counter()
for g in G:
    if g['net_color']=='black': c[divmod(g['moves'][0]['move'],9)]+=1
    else: c2[divmod(g['moves'][1]['move'],9)]+=1
print(c.most_common(8)); print(c2.most_common(8))
print(G[0]['moves'][0]['top5'])
```