```python
exec(open('/work/an.py').read())
G,boards=load_pass(7)
import numpy as np
tops=[]; chosen_rank=[]
for g in G:
    if g['opponent']!='strong': continue
    for m in g['moves']:
        if m['by']=='net':
            t=m['top5']; tops.append(t[0][1])
            mvs=[a for a,b in t]
            chosen_rank.append(mvs.index(m['move']) if m['move'] in mvs else 5)
tops=np.array(tops); cr=np.array(chosen_rank)
print('mean top prob',tops.mean(), 'frac top<0.5',(tops<0.5).mean(), 'frac top<0.3',(tops<0.3).mean())
print('chosen rank dist',np.bincount(cr,minlength=6)/len(cr))
```