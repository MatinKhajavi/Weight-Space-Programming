```python
exec(open('/work/builder27.py').read())
import json
prm=json.load(open('/work/prm_v27.json')); prm.update(cap=22.,esc=12.,sa=10.)
json.dump(prm,open('/work/prm_v29.json','w'))
W,B,Wo,bo=build(prm)
save_all(save,W,B,Wo,bo)
print(prm)
```