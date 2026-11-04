```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_7/'
G=json.load(open(d+'games.json'))
B=np.load(d+'boards.npy')
from collections import Counter
cnt=Counter()
for g in G:
    cnt[(g['opponent'],g['end'],g['result'],g['moves'][-1]['by'])]+=1
for k,v in sorted(cnt.items()): print(k,v)
def chain_bfs(b,q):
    col=b[q]; dist={q:0}; order=[q]; i=0
    while i<len(order):
        x=order[i]; i+=1
        for y in NB[x]:
            if b[y]==col and y not in dist: dist[y]=dist[x]+1; order.append(y)
    return dist
reps=[g for g in G if g['end']=='repetition' and g['result']==-1]
for g in reps:
    gi=g['game']; mv=g['moves']; n=len(mv)
    seq=[(m['by'][0], divmod(m['move'],9) if m['move']<81 else 'P', m['top5'][0][1] if m['by']=='net' else None) for m in mv[n-5:]]
    k=n-3; m=mv[k]; p=m['move']; b=B[gi,k-1]; col=1 if g['net_color']=='black' else -1
    info=[]
    if p<81:
      for y in NB[p]:
        if b[y]==-col:
            dist=chain_bfs(b,y); libs=set(z for x in dist for z in NB[x] if b[z]==0)
            info.append(('opp',len(dist),len(libs)))
        elif b[y]==col:
            s,l=chain(b,y); info.append(('own',len(s),len(l)))
    print(gi,g['opponent'],g['margin'],seq,info)
```