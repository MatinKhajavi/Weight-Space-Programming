```python
exec(open('/work/ana.py').read()); exec(open('/work/go.py').read())
G,B,V=load_pass(20)
def tob(b): return [1 if v==1 else (2 if v==-1 else 0) for v in b]
res=[]
for g in G:
    gi=g['game']; n=len(g['moves']); bd=tob(B[gi,n-1])
    dame=0
    for p in range(81):
        if bd[p]: continue
        cols={bd[q] for q in NB[p]}-{0}
        if cols=={1,2}: dame+=1
    res.append((g['result'],g['margin'],dame))
import numpy as np
r=np.array(res)
print('avg dame wins',r[r[:,0]>0,2].mean(),'losses',r[r[:,0]<0,2].mean())
print(sorted([(m,d) for s,m,d in res if s<0])[-12:])
```