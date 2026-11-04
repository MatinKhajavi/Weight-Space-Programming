```python
exec(open('/work/builder27.py').read())
import json
prm=json.load(open('/work/prm_v27.json'))
W,B,Wo,bo=build(prm)
save_all(save,W,B,Wo,bo)
print('restored v27', prm.get('passlead','none'), float(np.abs(Wo[494]).sum()))
```