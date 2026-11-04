```python
import json, numpy as np
exec(open('/work/goutil.py').read())
N=9
def onb(x): return 0<=x[0]<N and 0<=x[1]<N
pats=[]
for side in range(4):
    for t in range(1,8):
        if side==0: Bp=(0,t); e=(0,1); n_=(1,0)
        elif side==1: Bp=(8,t); e=(0,1); n_=(-1,0)
        elif side==2: Bp=(t,0); e=(1,0); n_=(0,1)
        else: Bp=(t,8); e=(1,0); n_=(0,-1)
        for s in (1,-1):
            Ap=(Bp[0]+s*e[0],Bp[1]+s*e[1]); Cp=(Bp[0]-s*e[0],Bp[1]-s*e[1])
            own=[X for X in ((Cp[0]-s*e[0],Cp[1]-s*e[1]),(Cp[0]+n_[0],Cp[1]+n_[1])) if onb(X)]
            opp=[X for X in ((Ap[0]+s*e[0],Ap[1]+s*e[1]),(Ap[0]+n_[0],Ap[1]+n_[1]),(Bp[0]+n_[0],Bp[1]+n_[1])) if onb(X)]
            pats.append((Ap,Bp,Cp,own,opp))
from collections import Counter
res=Counter()
for dd in ('/work/pass_14/','/work/pass_16/','/work/pass_10/'):
    G=json.load(open(dd+'games.json')); B=np.load(dd+'boards.npy')
    for g in G:
        gi=g['game']; col=1 if g['net_color']=='black' else -1
        mv=g['moves']
        for m in mv:
            if m['by']!='net': continue
            k=m['ply']
            if k<2: continue
            b=B[gi,k-1]
            for Ap,Bp,Cp,own,opp in pats:
                a,bb,cc=[x[0]*9+x[1] for x in (Ap,Bp,Cp)]
                if b[a]==0 and b[bb]==0 and b[cc]==-col and all(b[x[0]*9+x[1]]==col for x in own) and all(b[x[0]*9+x[1]]==-col for x in opp):
                    cyc = (mv[k-1]['move']==cc and mv[k-2]['move']==a)
                    res[(cyc, m['move']==bb)]+=1
print(res)
```