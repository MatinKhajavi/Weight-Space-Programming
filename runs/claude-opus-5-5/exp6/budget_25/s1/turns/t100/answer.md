```python
import json
g=dict(save=save); exec(open('/work/trunk11.py').read()+open('/work/head15.py').read(),g)
P=json.load(open('/work/P_v15.json')); P['w_attu']=3
g['save_all'](P); json.dump(P,open('/work/P_v16.json','w'))
open('/work/notes.txt','a').write("pass20 v15: strong 64% (best; 3 reps). v16 = v15 + w_attu 3 (P_v16.json). If not better revert to v15.\n")
print('ok')
```