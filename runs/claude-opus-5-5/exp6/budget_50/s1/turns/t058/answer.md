```python
import json, numpy as np
exec(open('/work/goutil.py').read())
from collections import Counter
for dd in ('/work/pass_10/','/work/pass_12/'):
    G=json.load(open(dd+'games.json')); B=np.load(dd+'boards.npy')
    tot=Counter()
    for g in G:
        gi=g['game']; col=1 if g['net_color']=='black' else -1; n=len(g['moves'])
        b=B[gi,n-1]
        seen=set(); weakopp=0; weakown=0
        for p in range(81):
            if b[p]!=0 and p not in seen:
                s,l=chain(b,p); seen|=s
                if len(l)<=1:
                    if b[p]==-col: weakopp+=len(s)
                    else: weakown+=len(s)
        firstpass=[m for m in g['moves'] if m['move']==81]
        tot['weakopp']+=weakopp; tot['weakown']+=weakown
        tot['net_first_pass' if firstpass and firstpass[0]['by']=='net' else 'opp_first_pass']+=1
        tot['len']+=n
        # first pass ply
        if firstpass: tot['fp_ply']+=firstpass[0]['ply']
    print(dd,{k:(v/len(G)) for k,v in tot.items()})
```