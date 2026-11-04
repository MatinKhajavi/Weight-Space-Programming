```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
G,boards=load_pass(2)
import collections
miss=[]
for gi,g in enumerate(G):
    mv=g['moves']; netcol = 1 if g['net_color']=='black' else -1
    for k,m in enumerate(mv):
        if m['by']!='net' or m['move']==81: continue
        b = (boards[gi,k-1] if k>0 else np.zeros(81)).astype(int)
        p=m['move']; la,cap=libs_after(b,p,netcol)
        ch,idx=chains(b)
        caps=[ (len(mem),list(libs)[0]) for col,mem,libs in ch if col==-netcol and len(libs)==1]
        if caps and cap==0:
            # was capture legal?
            legal=[c for c in caps if play(b,c[1],netcol)[0] is not None]
            if legal: miss.append((gi,k,legal,p,m['top5'][:3]))
print(len(miss))
c=collections.Counter(max(x[2])[0] for x in miss); print(c)
for x in miss[:12]:
    print(x)
```