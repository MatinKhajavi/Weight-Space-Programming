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
rng=np.random.default_rng(0)
idx=[(g,k) for g in range(B.shape[0]) for k in range(0,162,3) if V[g,k]]
for g,k in idx:
    b=B[g,k]
    for q in range(81):
        if b[q]==0: continue
        if not any(b[y]==0 for y in NB[q]): continue
        dist=chain_bfs(b,q)
        libs=set(y for x in dist for y in NB[x] if b[y]==0)
        true1 = len(libs)==1
        # evidence within hops H
        for H in (3,4,99):
            st=[x for x in dist if dist[x]<=H]
            e={x:sum(1 for y in NB[x] if b[y]==0) for x in st}
            ra=any(v>=2 for v in e.values())
            par=set((x//9+x%9)%2 for x in st if e[x]>=1)
            rp = len(par)==2
            rb=any(e[x]>=1 and any(y in e and e[y]>=1 for y in NB[x]) for x in st)
            for name,ev in (('a+b',ra or rb),('a+par',ra or rp)):
                res[(H,name,true1,ev)]+=1
for H in (3,4,99):
    for name in ('a+b','a+par'):
        fa=res[(H,name,False,False)]; tot2=res[(H,name,False,False)]+res[(H,name,False,True)]
        bad=res[(H,name,True,True)]
        print(H,name,'false-atari',fa,'of',tot2,'multi-lib stones; unsound',bad,' true atari',res[(H,name,True,False)])
```