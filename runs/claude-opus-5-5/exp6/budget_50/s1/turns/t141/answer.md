```python
exec(open('/work/builder24.py').read())
import json
prm=json.load(open('/work/prm_v24.json')); prm['ownadj']=0.0
json.dump(prm,open('/work/prm_v26.json','w'))
W,B,Wo,bo=build(prm)
save_all(save,W,B,Wo,bo)
print(prm)
```