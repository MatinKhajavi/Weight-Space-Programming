```python
import json
g=dict(save=save); exec(open('/work/trunk11.py').read()+open('/work/head12.py').read(),g)
P=json.load(open('/work/P_v12.json'))
P.update(own_a=0.2, own_d=0.4, own_j=0.2, own_k=0.1)
g['save_all'](P); json.dump(P,open('/work/P_v13.json','w'))
open('/work/notes.txt','a').write("pass17 v12: strong 59.4% (3 reps). losses = big groups dying (lost 24 stones avg). v13 = v12 + connected patterns own_a .2 own_d .4 own_j .2 own_k .1 (P_v13.json)\n")
print('ok')
```