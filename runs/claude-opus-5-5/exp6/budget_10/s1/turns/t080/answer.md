```python
exec(open('/work/an.py').read())
G,boards=load_pass(8)
import numpy as np
tops=[]; cr=[]
for g in G:
    if g['opponent']!='strong': continue
    for m in g['moves']:
        if m['by']=='net':
            t=m['top5']; tops.append(t[0][1]); mvs=[a for a,b in t]
            cr.append(mvs.index(m['move']) if m['move'] in mvs else 5)
tops=np.array(tops); cr=np.array(cr)
print('mean top',tops.mean(),'frac<0.5',(tops<0.5).mean())
print(np.bincount(cr,minlength=6)/len(cr))
```