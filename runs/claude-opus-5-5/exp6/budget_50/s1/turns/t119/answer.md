```python
exec(open('/work/builder19.py').read())
import json
prm=json.load(open('/work/prm_v19.json')); prm['kw']=2.5
json.dump(prm,open('/work/prm_v20.json','w'))
W,B,Wo,bo=build(prm)
save_all(save,W,B,Wo,bo)
print(prm)
```