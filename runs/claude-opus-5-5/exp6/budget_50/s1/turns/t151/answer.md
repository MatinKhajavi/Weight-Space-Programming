```python
exec(open('/work/builder27.py').read())
import json
prm=json.load(open('/work/prm_v27.json')); prm.update(jump=1.5,knight=0.8)
json.dump(prm,open('/work/prm_v28.json','w'))
W,B,Wo,bo=build(prm)
save_all(save,W,B,Wo,bo)
print(prm['jump'],prm['knight'])
```