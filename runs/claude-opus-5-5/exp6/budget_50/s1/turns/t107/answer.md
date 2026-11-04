```python
exec(open('/work/builder15.py').read())
import json
prm=json.load(open('/work/prm_v17.json'))
sc=1.5
for k in ('contact','ownadj','owndiag','oppdiag','ot','tengen','cap','esc','sa','eye','cyc','pass'): prm[k]*=sc
prm['lineval']=[x*sc for x in prm['lineval']]
json.dump(prm,open('/work/prm_v18.json','w'))
W,B,Wo,bo=build(prm)
save_all(save,W,B,Wo,bo)
print(prm)
```