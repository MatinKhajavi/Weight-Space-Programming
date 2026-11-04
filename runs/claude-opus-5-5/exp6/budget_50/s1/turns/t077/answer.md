```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_16/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
for gi in (1,91,99,72):
    g=G[gi]; mv=g['moves']; n=len(mv)
    lastb=B[gi,n-1]
    rep=[k for k in range(n-2,max(-1,n-10),-1) if (B[gi,k]==lastb).all()]
    k=rep[0]; b=B[gi,k]
    trig=mv[k+1]
    print(gi,g['net_color'],'trigger',divmod(trig['move'],9),trig['top5'][:3])
    rows=to_strings(b); r0,c0=divmod(trig['move'],9)
    for rr in range(max(0,r0-3),min(9,r0+4)): print('    ',rows[rr])
```