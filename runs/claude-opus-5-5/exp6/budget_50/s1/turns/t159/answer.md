```python
exec(open('/work/builder27.py').read())
import json
prm=json.load(open('/work/prm_v27.json')); prm.update(ot=8.)
json.dump(prm,open('/work/prm_v30.json','w'))
W,B,Wo,bo=build(prm)
save_all(save,W,B,Wo,bo)
print(prm['ot'],prm['cap'],prm['esc'],prm['sa'])
```