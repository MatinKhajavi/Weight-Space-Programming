```python
exec(open('/work/ana.py').read()); exec(open('/work/go.py').read())
G,B,V=load_pass(13)
def tob(b): return [1 if v==1 else (2 if v==-1 else 0) for v in b]
missed=[]; total=0
for g in G:
    gi=g['game']; mv=g['moves']; me=1 if g['net_color']=='black' else 2
    for k,m in enumerate(mv):
        if m['by']!='net' or k==0: continue
        bd=tob(B[gi,k-1])
        # capturing moves available
        caps={}
        for p in range(81):
            if bd[p]: continue
            nb2=play(bd,p,me)
            if nb2 is None: continue
            c=sum(1 for q in range(81) if bd[q]==3-me and nb2[q]==0)
            if c>0: caps[p]=c
        if caps:
            total+=1
            if m['move'] not in caps:
                missed.append((gi,k,caps,m['move'],m['top5'][:2]))
print(total,len(missed))
for x in missed[:12]: print(x)
```