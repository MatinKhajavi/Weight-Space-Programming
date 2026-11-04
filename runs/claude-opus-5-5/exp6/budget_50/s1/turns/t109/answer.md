```python
import json, numpy as np
exec(open('/work/goutil.py').read())
from collections import Counter
cat=Counter()
for dd in ('/work/pass_24/','/work/pass_25/'):
    G=json.load(open(dd+'games.json')); B=np.load(dd+'boards.npy')
    for g in G:
        if g['end']!='repetition' or g['result']!=-1: continue
        gi=g['game']; mv=g['moves']; n=len(mv); col=1 if g['net_color']=='black' else -1
        # find first net move in cycle: assume cycle length up to 8 plies; find k where board after ply k equals final
        lastb=B[gi,n-1]
        rep=[k for k in range(n-2,max(-1,n-10),-1) if (B[gi,k]==lastb).all()]
        if not rep:
            cat['unknown']+=1; print(dd[-8:],gi,'unknown', [(m['by'][0],divmod(m['move'],9) if m['move']<81 else 'P') for m in mv[-6:]]); continue
        seq=[(m['by'][0],m['move']) for m in mv[rep[0]+1:]]
        L=len(seq); haspass=any(x[1]==81 for x in seq)
        # trigger = first net move
        trig=[m for m in mv[rep[0]+1:] if m['by']=='net'][0]
        k=trig['ply']; b=B[gi,k-1]; p=trig['move']
        nb,cap=play(b,p,col) if p<81 else (b,0)
        libs=len(chain(nb,p)[1]) if (p<81 and nb is not None) else -1
        cat[(L,haspass,'trigcap' if cap else 'nocap',libs)]+=1
        print(dd[-8:],gi,g['margin'],L,haspass,'trig',divmod(p,9),trig['top5'][0],'cap',cap,'libs after',libs)
print(cat)
```