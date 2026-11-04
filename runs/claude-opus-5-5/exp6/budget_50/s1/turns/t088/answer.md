```python
exec(open('/work/builder9.py').read())
import json
prm=json.load(open('/work/prm_v9.json'))
sc=2.0
for k in ('contact','ownadj','owndiag','oppdiag','ot','tengen'): prm[k]*=sc
prm['lineval']=[x*sc for x in prm['lineval']]
json.dump(prm,open('/work/prm_v14.json','w'))
W,B,Wo,bo=build(prm)
save_all(save,W,B,Wo,bo)
print(prm)
```