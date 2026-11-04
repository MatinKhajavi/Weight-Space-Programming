```python
import json
g=dict(save=save); exec(open('/work/trunk11.py').read()+open('/work/head15.py').read(),g)
P=json.load(open('/work/P_v15.json'))
g['save_all'](P)
Ws,bs=g['build_all'](P)
print('saved matches v15:',all(np.array_equal(load(k+'.W'),Ws[k]) and np.array_equal(load(k+'.b'),bs[k]) for k in ['L1','L2','L3','L4','L5','L6','out']))
open('/work/notes.txt','a').write("pass23 v18 (scale4): 62.5% ~ same. FINAL = v15 restored (P_v15.json).\n")
```