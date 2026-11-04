```python
exec(open('/work/ana.py').read()); exec(open('/work/go.py').read())
from collections import Counter
def tob(b): return [1 if v==1 else (2 if v==-1 else 0) for v in b]
cnt=Counter()
for k in (10,13):
    G,B,V=load_pass(k)
    for g in G:
        gi=g['game']; mv=g['moves']; me=1 if g['net_color']=='black' else 2
        for kk,m in enumerate(mv):
            if m['by']!='net' or kk==0: continue
            bd=tob(B[gi,kk-1])
            for p in range(81):
                if bd[p]: continue
                nb2=play(bd,p,me)
                if nb2 is None: continue
                c=sum(1 for q in range(81) if bd[q]==3-me and nb2[q]==0)
                if c>=3:
                    e=sum(1 for q in NB[p] if bd[q]==0)
                    nown=sum(1 for q in NB[p] if bd[q]==me)
                    own3=any(bd[q]==me and len(group(bd,q)[1])>=3 for q in NB[p])
                    if e<=1 and not own3:
                        cnt[(k,'nown=0' if nown==0 else 'nown>0')]+=1
                    break
print(cnt)
```