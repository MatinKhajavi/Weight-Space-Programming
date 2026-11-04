```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_2/'
B=np.load(d+'boards.npy'); V=np.load(d+'valid.npy')
from collections import Counter
def chain_bfs(b,q):
    col=b[q]; dist={q:0}; order=[q]; i=0
    while i<len(order):
        x=order[i]; i+=1
        for y in NB[x]:
            if b[y]==col and y not in dist: dist[y]=dist[x]+1; order.append(y)
    return dist
res=Counter()
idx=[(g,k) for g in range(B.shape[0]) for k in range(0,162,3) if V[g,k]]
for g,k in idx:
    b=B[g,k]
    for q in range(81):
        if b[q]==0 or not any(b[y]==0 for y in NB[q]): continue
        dist=chain_bfs(b,q)
        libs=set(y for x in dist for y in NB[x] if b[y]==0)
        t1=len(libs)==1
        if t1: continue
        e={x:sum(1 for y in NB[x] if b[y]==0) for x in dist}
        # rule a source at distance d ; rule b source (pair) at distance of nearer of pair
        da=min([dist[x] for x in dist if e[x]>=2],default=99)
        db=min([dist[x] for x in dist if e[x]>=1 and any(y in dist and e[y]>=1 for y in NB[x])],default=99)
        # rule b computed in L2 from L1 -> effectively one hop later: evidence at x needs both x and neighbor => source at x
        for H in (4,5):
            ab = da<=H or db<=H-1   # rule-b needs one layer to form
            par=set((x//9+x%9)%2 for x in dist if dist[x]<=H and e[x]>=1)
            ap = da<=H or len(par)==2
            res[(H,'ab',ab)]+=1; res[(H,'ap',ap)]+=1
print(sorted(res.items()))
```