```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_9/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
reps=[g for g in G if g['end']=='repetition']
for g in reps:
    gi=g['game']; mv=g['moves']; n=len(mv)
    lastb=B[gi,n-1]
    rep=[k for k in range(n-2,max(-1,n-10),-1) if (B[gi,k]==lastb).all()]
    k0=rep[0]+1
    trig=[m for m in mv[k0:] if m['by']=='net'][0]
    k=trig['ply']; b=B[gi,k-1]; p=trig['move']
    info=[]
    if p<81:
        for y in NB[p]:
            if b[y]!=0:
                s,l=chain(b,y); info.append(('own' if (b[y]==(1 if g['net_color']=='black' else -1)) else 'opp',len(s),len(l)))
    print(gi,g['result'],divmod(p,9) if p<81 else 'P',trig['top5'][:2],info)
```