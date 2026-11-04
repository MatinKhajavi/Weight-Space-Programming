```python
exec(open('/work/ana.py').read()); exec(open('/work/go.py').read())
from collections import Counter
def tob(b): return [1 if v==1 else (2 if v==-1 else 0) for v in b]
G,B,V=load_pass(13)
cnt=Counter(); ex=[]
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
            if c>=2:
                e=sum(1 for q in NB[p] if bd[q]==0)
                own3=any(bd[q]==me and len(group(bd,q)[1])>=3 for q in NB[p])
                gsz=max(len(group(bd,q)[0]) for q in NB[p] if bd[q]==3-me and nb2[q]==0)
                selfflag = (e<=1 and not own3)
                took = m['move']==p
                cnt[(selfflag, 'sz>2' if gsz>2 else 'sz2', took)]+=1
                if not took and not selfflag and len(ex)<8: ex.append((gi,kk,divmod(p,9),c,gsz,e,m['move'],m['top5'][:2]))
                break
for k,v in sorted(cnt.items()): print(k,v)
for x in ex: print(x)
```