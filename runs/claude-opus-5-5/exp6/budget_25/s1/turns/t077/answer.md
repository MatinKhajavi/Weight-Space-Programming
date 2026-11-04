```python
import json
g=dict(save=save); exec(open('/work/trunk7.py').read()+open('/work/head10.py').read(),g)
P=json.load(open('/work/P_v10.json')); P['center_bonus']=2.5
g['save_all'](P); json.dump(P,open('/work/P_v10.json','w'))
print(P)
```