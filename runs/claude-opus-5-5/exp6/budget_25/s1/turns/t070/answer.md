```python
exec(open('/work/ana.py').read()); exec(open('/work/go.py').read())
from collections import Counter
def tob(b): return [1 if v==1 else (2 if v==-1 else 0) for v in b]
def analyze(k):
    G,B,V=load_pass(k)
    cnt=Counter()
    for g in G:
        gi=g['game']; mv=g['moves']; me=1 if g['net_color']=='black' else 2
        for kk,m in enumerate(mv):
            if m['by']!='net' or kk==0: continue
            bd=tob(B[gi,kk-1])
            best=None
            for p in range(81):
                if bd[p]: continue
                nb2=play(bd,p,me)
                if nb2 is None: continue
                c=sum(1 for q in range(81) if bd[q]==3-me and nb2[q]==0)
                if c>0:
                    gg,l=group(nb2,p)
                    typ=('ko' if (c==1 and len(l)==1 and len(gg)==1) else ('c1' if c==1 else 'cbig'))
                    if best is None or c>best[1]: best=(p,c,typ)
            if best:
                took = m['move']!=81 and tob(B[gi,kk])[best[0]]==me and any(tob(B[gi,kk-1])[q]==3-me and tob(B[gi,kk])[q]==0 for q in range(81))
                cnt[(best[2],'took' if took else 'missed')]+=1
    return cnt
for k in (10,13):
    print(k, sorted(analyze(k).items()))
```