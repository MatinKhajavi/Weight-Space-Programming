```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_14/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy'); V=np.load(d+'valid.npy')
rows=[]
for g in G:
    gi=g['game']; col=1 if g['net_color']=='black' else -1; n=len(g['moves'])
    traj=[]
    for k in (30,50,70,90,n-1):
        k=min(k,n-1)
        s=area_score(B[gi,k])-7.5
        traj.append(round(col*s,1))
    rows.append((g['result'],g['net_color'],traj,g['margin']))
import collections
for res in (1,-1):
    arr=np.array([r[2] for r in rows if r[0]==res])
    print(res, len(arr), 'mean traj', arr.mean(0).round(1), 'median', np.median(arr,0))
# losses with large lead at ply 70
print([r for r in rows if r[0]==-1][:20])
```