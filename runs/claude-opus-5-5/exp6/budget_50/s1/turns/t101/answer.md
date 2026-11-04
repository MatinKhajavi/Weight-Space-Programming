```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_23/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
for gi in (32,14,117):
    g=G[gi]; mv=g['moves']; n=len(mv)
    lastb=B[gi,n-1]
    rep=[k for k in range(n-2,max(-1,n-10),-1) if (B[gi,k]==lastb).all()]
    print('game',gi,g['net_color'],'rep of',rep)
    for k in range(max(0,n-10),n):
        m=mv[k]
        print('  ',k,m['by'][0],divmod(m['move'],9) if m['move']<81 else 'P', m['top5'][:3] if m['by']=='net' else '')
    k0=rep[0] if rep else n-3
    for s in to_strings(B[gi,k0]): print('     ',s)
```