```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
G,boards=load_pass(3)
import collections
cnt=collections.Counter()
for gi,g in enumerate(G):
    if g['end']=='repetition':
        mv=g['moves']; last=mv[-1]
        cnt[(g['opponent'],last['by'],g['result'])]+=1
print(cnt)
for gi,g in enumerate(G):
    if g['end']=='repetition' and g['result']<0:
        mv=g['moves']; n=len(mv)
        print(gi,g['opponent'],g['net_color'],g['margin'],[(m['by'][0],m['move'],m.get('top5',[[0,0]])[0]) for m in mv[-6:]])
```