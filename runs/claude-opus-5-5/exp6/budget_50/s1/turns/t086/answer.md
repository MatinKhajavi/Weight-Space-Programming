```python
exec(open('/work/builder9.py').read())
import json
prm=json.load(open('/work/prm_v9.json'))
W,B,Wo,bo=build(prm)
save_all(save,W,B,Wo,bo)
print(prm)
```