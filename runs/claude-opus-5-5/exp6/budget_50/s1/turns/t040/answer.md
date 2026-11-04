```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_7/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
meta=json.load(open('/work/testpos2_meta.json'))
pos=[]
for gi,k,p,cap in meta:
    g=G[gi]; b=B[gi,k-1]
    pos.append({"board":to_strings(b),"to_move":g['net_color']})
json.dump(pos,open('/work/testpos2b.json','w'))
print(json.dumps({"positions":pos,"activations":False}))
```