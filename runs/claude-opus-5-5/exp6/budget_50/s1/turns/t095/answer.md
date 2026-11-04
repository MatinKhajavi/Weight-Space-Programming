```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_21/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
reps=[g for g in G if g['end']=='repetition' and g['result']==-1]
shown=0
for g in reps:
    gi=g['game']; mv=g['moves']; n=len(mv)
    lastb=B[gi,n-1]
    rep=[k for k in range(n-2,max(-1,n-10),-1) if (B[gi,k]==lastb).all()]
    if not rep: continue
    k0=rep[0]
    print('game',gi,g['net_color'],'margin',g['margin'])
    for k in range(k0,n):
        m=mv[k] if k>k0 else None
        lab = '' if k==k0 else (m['by'][0]+' '+(str(divmod(m['move'],9)) if m['move']<81 else 'P')+' '+(str(m['top5'][:3]) if m['by']=='net' else ''))
        print('  ply',k,lab)
    # print board region diff
    rows0=to_strings(B[gi,k0])
    for s in rows0: print('     ',s)
    shown+=1
    if shown>=3: break
```