```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_10/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def chain_bfs(b,q):
    col=b[q]; dist={q:0}; order=[q]; i=0
    while i<len(order):
        x=order[i]; i+=1
        for y in NB[x]:
            if b[y]==col and y not in dist: dist[y]=dist[x]+1; order.append(y)
    return dist
reps=[g for g in G if g['end']=='repetition']
for g in reps:
    gi=g['game']; mv=g['moves']; n=len(mv)
    last=mv[-1]
    # find cycle start: the repeated board index
    lastb=B[gi,n-1]
    rep=[k for k in range(n-2,max(-1,n-10),-1) if (B[gi,k]==lastb).all()]
    seq=[(m['by'][0], divmod(m['move'],9) if m['move']<81 else 'P', round(m['top5'][0][1],2) if m['by']=='net' else None) for m in mv[rep[0]:]] if rep else None
    print(gi,g['net_color'],g['result'],g['margin'],'repeat of ply',rep[:1],seq)
```