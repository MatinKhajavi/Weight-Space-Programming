```python
g=dict(save=save); exec(open('/work/trunk3.py').read()+open('/work/head3.py').read(),g)
P=dict(g['P3']); P.update(w_cap=4,w_capS=0.5,w_self=-10)
g['save_all'](P)
import json; json.dump(P,open('/work/P_v3b.json','w'))
open('/work/notes.txt','a').write("v3b: P3 + w_cap=4,w_capS=0.5,w_self=-10 (saved P_v3b.json). pos-pass4 showed big-group false atari still -> capture bonus too high\n")
print('ok')
```