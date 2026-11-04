```python
import json, numpy as np
exec(open('/work/goutil.py').read())
from collections import Counter
D=np.array([[abs(a//9-c//9)+abs(a%9-c%9) for c in range(81)] for a in range(81)])
for dd in ('/work/pass_7/',):
    G=json.load(open(dd+'games.json')); B=np.load(dd+'boards.npy')
    capsz=Counter(); false=Counter(); 
    for g in G:
        if g['opponent']!='strong': continue
        gi=g['game']; col=1 if g['net_color']=='black' else -1
        for m in g['moves']:
            if m['by']!='net' or m['move']==81: continue
            k=m['ply']; prev=B[gi,k-1] if k>0 else np.zeros(81,int); p=m['move']
            nb,cap=play(prev,p,col)
            if cap>0: capsz[min(cap,10)]+=1
            elif m['top5'][0][0]==p and m['top5'][0][1]>0.9:
                opp=[y for y in NB[p] if prev[y]==-col]
                if opp:
                    s,l=chain(prev,opp[0])
                    m2=int(((prev==-col)&(D[opp[0]]<=2)&(D[opp[0]]>0)).sum())
                    false[(min(len(s),10),m2)]+=1
    print('captures by size',sorted(capsz.items()))
    print('confident non-captures adjacent to opp (size,m2)',sorted(false.items()))
```