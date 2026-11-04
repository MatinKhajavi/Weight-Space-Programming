```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
import collections
G,boards=load_pass(5)
cases=[]
for gi,g in enumerate(G):
    if g['opponent']!='strong': continue
    netcol = 1 if g['net_color']=='black' else -1
    mv=g['moves']
    for kk,m in enumerate(mv):
        if m['by']!='net' or m['move']==81: continue
        b=boards[gi,kk-1].astype(int); p=m['move']
        la,cap=libs_after(b,p,netcol)
        if la==1:
            ch,idx=chains(b)
            e=sum(1 for q in NBR[p] if b[q]==0)
            own=[(len(ch[idx[q]][1]),len(ch[idx[q]][2])) for q in NBR[p] if b[q]==netcol]
            opp=[(len(ch[idx[q]][1]),len(ch[idx[q]][2])) for q in NBR[p] if b[q]==-netcol]
            cases.append((gi,kk,p,cap,e,own,opp,m['top5'][0]))
print(len(cases))
c=collections.Counter((x[3]>0, x[4], len(x[5])) for x in cases); print(c)
for x in cases[:15]: print(x)
```