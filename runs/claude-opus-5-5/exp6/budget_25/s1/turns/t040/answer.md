```python
exec(open('/work/ana.py').read())
from collections import Counter
G,B,V=load_pass(6)
ls=[g for g in G if g['opponent']=='strong' and g['result']<0]
print(sorted(g['margin'] for g in ls))
for g in ls[:3]:
    gi=g['game']; mv=g['moves']; n=len(mv)
    fp=[m['ply'] for m in mv if m['by']=='net' and m['move']==81]
    print(gi,g['net_color'],g['margin'],'net passes at',fp[:10],'len',n)
    k=fp[0]
    print(sb(B[gi,k-1]))
    print([(divmod(a,9) if a<81 else 'P',p) for a,p in mv[k]['top5']])
```