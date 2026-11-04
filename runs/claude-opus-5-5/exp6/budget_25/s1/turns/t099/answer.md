```python
exec(open('/work/ana.py').read()); exec(open('/work/go.py').read())
from collections import defaultdict
G,B,V=load_pass(20)
def tob(b): return [1 if v==1 else (2 if v==-1 else 0) for v in b]
games=defaultdict(int)
for g in G:
    gi=g['game']; me=1 if g['net_color']=='black' else 2
    for k,m in enumerate(g['moves']):
        if m['by']=='net' and m['move']==81:
            bd=tob(B[gi,k-1])
            for p in range(81):
                if bd[p]: continue
                nb2=play(bd,p,me)
                if nb2 is None: continue
                c=sum(1 for q in range(81) if bd[q]==3-me and nb2[q]==0)
                if c>=2:
                    games[gi]=max(games[gi],c)
for gi,c in sorted(games.items()):
    g=G[gi]; print(gi,'maxcap',c,'result',g['result'],'margin',g['margin'])
```