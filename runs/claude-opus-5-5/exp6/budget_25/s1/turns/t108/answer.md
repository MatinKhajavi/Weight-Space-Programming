```python
import json
g=dict(save=save); exec(open('/work/trunk11.py').read()+open('/work/head15.py').read(),g)
P=json.load(open('/work/P_v15.json'))
print(P['w_ko'],P['w_attu'],P['scale'])
g['save_all'](P)
Ws,bs=g['build_all'](P)
ok=all(np.array_equal(load(k+'.W'),Ws[k]) and np.array_equal(load(k+'.b'),bs[k]) for k in ['L1','L2','L3','L4','L5','L6','out'])
print('saved matches v15:',ok)
open('/work/notes.txt','a').write("pass22 v17 (w_ko -35): strong 50%, weak 94%, random 100% -> revert. Restored v15 as current.\n")
```