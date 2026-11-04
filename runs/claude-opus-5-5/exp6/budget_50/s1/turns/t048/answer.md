```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_10/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
for gi,k in ((23,59),(65,90),(36,89)):
    b=B[gi,k]; g=G[gi]; col=1 if g['net_color']=='black' else -1
    print(gi,'net',g['net_color'],'next net move',g['moves'][k+1]['top5'][:3])
    rows=to_strings(b)
    for s in rows: print('   ',s)
    p=g['moves'][k+1]['move']
    for y in NB[p]:
        if b[y]!=0:
            s,l=chain(b,y); print('  nbr',divmod(y,9),'col',b[y],'size',len(s),'libs',sorted(divmod(z,9) for z in l))
```