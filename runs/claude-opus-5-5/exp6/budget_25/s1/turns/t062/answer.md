```python
t7=open('/work/trunk7.py').read()
t7=t7.replace("""        u=SAFE3+p; b[u]=-12/1024-16384; W[OWN+p,u]+=16384
        for q in [p]+NB[p]: W[VO+q,u]+=1/1024""","""        u=SAFE3+p; b[u]=-4/1024; W[VO+p,u]+=1/1024""")
assert "W[VO+p,u]+=1/1024" in t7
open('/work/trunk7.py','w').write(t7)
g=dict(save=save); exec(t7+open('/work/head7.py').read(),g)
import json; P=json.load(open('/work/P_v7.json'))
Ws,bs=g['build_all'](P)
print({k:float(np.abs(Ws[k]).max()) for k in Ws})
g['save_all'](P)
open('/work/notes.txt','a').write("v7 fix: SAFE3=relu((VO(p)-4)/1024) no nbr hop, no gate.\n")
```