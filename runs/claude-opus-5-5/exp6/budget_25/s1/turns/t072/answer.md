```python
exec(open('/work/ana.py').read()); exec(open('/work/go.py').read())
import numpy as np
from collections import Counter
def tob(b): return [1 if v==1 else (2 if v==-1 else 0) for v in b]
def mass(bd,q,col):
    v={q:1}
    # (I+A)^3 applied to ones restricted to group: compute vector over group
    g,_=group(bd,q); g=list(g); idx={s:i for i,s in enumerate(g)}
    A=np.eye(len(g))
    for s in g:
        for t in NB[s]:
            if t in idx: A[idx[s],idx[t]]=1
    return int((np.linalg.matrix_power(A,3)@np.ones(len(g)))[idx[q]])
real=[]; bigsafe=[]
for k in (10,13):
    G,B,V=load_pass(k)
    for g in G[::2]:
        gi=g['game']; me=1 if g['net_color']=='black' else 2
        for kk in range(20,len(g['moves']),3):
            bd=tob(B[gi,kk-1]); 
            for p in range(81):
                if bd[p]: continue
                for q in NB[p]:
                    if bd[q]==3-me:
                        gg,l=group(bd,q)
                        if len(l)==1 and len(gg)>=3: real.append((mass(bd,q,3-me),len(gg)))
                        if len(l)==2 and len(gg)>=8: bigsafe.append((mass(bd,q,3-me),len(gg)))
real=np.array(real); bigsafe=np.array(bigsafe)
print(len(real),len(bigsafe))
for K in (13,18,22,26,30,35,40,50):
    print(K, 'real exempt %.2f'%np.mean(real[:,0]<K), 'big2lib exempt %.2f'%np.mean(bigsafe[:,0]<K))
```