```python
import json
g=dict(save=save); exec(open('/work/trunk11.py').read()+open('/work/head15.py').read(),g)
P=json.load(open('/work/P_v15.json')); P['w_ko']=-35
g['save_all'](P); json.dump(P,open('/work/P_v17.json','w'))
open('/work/notes.txt','a').write("pass21 v16 (w_attu3): 58.6%, 10 reps (double-ko 5-cycles: false big U1 bonus overrides KO). v17 = v15 + w_ko -35 (P_v17.json)\n")
print(P)
```