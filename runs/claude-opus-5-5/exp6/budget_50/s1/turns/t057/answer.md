```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_12/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
pos=[]; meta=[]
for gi in (9,18,24,39,73,32):
    g=G[gi]; mv=g['moves']; n=len(mv)
    lastb=B[gi,n-1]
    rep=[k for k in range(n-2,max(-1,n-10),-1) if (B[gi,k]==lastb).all()]
    k=rep[0]  # board before trigger
    b=B[gi,k]
    pos.append({"board":to_strings(b),"to_move":g['net_color']}); meta.append((gi,k,mv[k+1]['move']))
    print(gi, g['net_color']); 
    for s in to_strings(b): print('   ',s)
json.dump(meta,open('/work/testpos3_meta.json','w')); json.dump(pos,open('/work/testpos3.json','w'))
print(json.dumps({"positions":pos,"activations":True}))
```