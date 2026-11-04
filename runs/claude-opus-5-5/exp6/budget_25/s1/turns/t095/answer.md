```python
h12=open('/work/head12.py').read()
h15=h12.replace("W[OWN+q,u]-=2; W[AXS+q,u]-=1/6","W[OWN+q,u]-=2; W[AXS+q,u]-=1/3")
assert h15!=h12
open('/work/head15.py','w').write(h15)
g=dict(save=save); exec(open('/work/trunk11.py').read()+h15,g)
import json; P=json.load(open('/work/P_v14.json'))
g['save_all'](P); json.dump(P,open('/work/P_v15.json','w'))
open('/work/notes.txt','a').write("pass19 v14 (scale 3): 61% but 11 reps (ko 5-cycles: EXL fired from false big atari nbrs on ko capture). v15: head15 EXL AXS coef 1/3. P_v15.json = P_v14\n")
print('ok')
```