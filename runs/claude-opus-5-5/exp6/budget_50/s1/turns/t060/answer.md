```python
exec(open('/work/builder5.py').read())
import json
prm=json.load(open('/work/prm_v5.json'))
W,B,Wo,bo=build(prm)
save_all(save,W,B,Wo,bo)
pos=json.load(open('/work/testpos3.json'))
print(json.dumps({"positions":pos,"activations":True}))
```