```python
import json
g=dict(save=save); exec(open('/work/trunk11.py').read()+open('/work/head15.py').read(),g)
P=json.load(open('/work/P_v15.json')); P['scale']=4.0
g['save_all'](P); json.dump(P,open('/work/P_v18.json','w'))
open('/work/notes.txt','a').write("v18 = v15 with scale 4.0 (P_v18.json); adopt only if >=64%, else restore v15.\n")
print('ok')
```