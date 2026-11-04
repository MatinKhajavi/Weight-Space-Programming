```python
exec(open('/work/builder15.py').read())
import json
prm=json.load(open('/work/prm_v17.json')); prm['pass']=-11.
json.dump(prm,open('/work/prm_v17.json','w'))
W,B,Wo,bo=build(prm)
save_all(save,W,B,Wo,bo)
print(bo[81])
```