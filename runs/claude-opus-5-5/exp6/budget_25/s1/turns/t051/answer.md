```python
import json
g=dict(save=save); exec(open('/work/trunk5.py').read()+open('/work/head5.py').read(),g)
P=json.load(open('/work/P_v5.json')); P['w_save']=0
g['save_all'](P); json.dump(P,open('/work/P_v5b.json','w'))
open('/work/notes.txt','a').write("pass8 v5: strong 51% (12 rep losses: SAVE unit fired on false big atari via U1 sum>1). v5b: w_save=0 (P_v5b.json)\n")
print('ok')
```