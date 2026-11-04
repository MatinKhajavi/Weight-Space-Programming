```python
import json
g=dict(save=save); exec(open('/work/trunk11.py').read()+open('/work/head12.py').read(),g)
P=json.load(open('/work/P_v12.json')); P['scale']=3.0
g['save_all'](P); json.dump(P,open('/work/P_v14.json','w'))
open('/work/notes.txt','a').write("pass18 v13 (connected patterns): 51.6% -> revert. v14 = v12 with scale 3.0 (P_v14.json)\n")
print('ok')
```