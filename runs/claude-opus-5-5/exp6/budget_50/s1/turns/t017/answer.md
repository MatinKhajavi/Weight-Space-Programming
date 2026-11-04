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
        for H in (4,5,6):
            st=[x for x in dist if dist[x]<=H]
            e={x:sum(1 for y in NB[x] if b[y]==0) for x in st}
            ra=any(v>=2 for v in e.values())
            par=set((x//9+x%9)%2 for x in st if e[x]>=1)
            ev= ra or len(par)==2
            if not ev:
                res[(H,t1)]+=1
print({k:v for k,v in sorted(res.items())})
```