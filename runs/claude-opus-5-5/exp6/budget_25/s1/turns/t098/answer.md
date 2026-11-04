```python
exec(open('/work/ana.py').read()); exec(open('/work/go.py').read())
from collections import Counter
G,B,V=load_pass(20)
def tob(b): return [1 if v==1 else (2 if v==-1 else 0) for v in b]
cnt=Counter(); ex=[]
for g in G:
    gi=g['game']; me=1 if g['net_color']=='black' else 2
    for k,m in enumerate(g['moves']):
        if m['by']=='net' and m['move']==81:
            bd=tob(B[gi,k-1]); best=None
            for p in range(81):
                if bd[p]: continue
                nb2=play(bd,p,me)
                if nb2 is None: continue
                c=sum(1 for q in range(81) if bd[q]==3-me and nb2[q]==0)
                if c:
                    gg,l=group(nb2,p)
                    nown=sum(1 for q in NB[p] if bd[q]==me); e=sum(1 for q in NB[p] if bd[q]==0)
                    typ='ko' if (c==1 and len(gg)==1 and len(l)==1) else ('single' if c==1 else 'big')
                    key=(typ,'nown0' if nown==0 else 'nown>0', 'e%d'%e)
                    if best is None or c>best[0]: best=(c,key,p)
            if best:
                cnt[best[1]]+=1
                if best[1][0]!='ko' and len(ex)<6: ex.append((gi,k,best))
for k,v in sorted(cnt.items(), key=lambda x:-x[1]): print(k,v)
print(ex)
```