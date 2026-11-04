```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
import collections
G,boards=load_pass(4)
shown=0; reasons=collections.Counter()
for gi,g in enumerate(G):
    if g['opponent']!='strong': continue
    netcol = 1 if g['net_color']=='black' else -1
    for kk,m in enumerate(g['moves']):
        if m['by']!='net' or m['move']==81: continue
        b=boards[gi,kk-1].astype(int)
        ch,idx=chains(b)
        p=m['move']
        la_p,cap_p=libs_after(b,p,netcol)
        for col,mem,libs in ch:
            if col==netcol and len(libs)==1 and len(mem)<3:
                lib=list(libs)[0]; la,cap=libs_after(b,lib,netcol)
                if la and la>=2 and p!=lib:
                    iseye = all(b[q]==netcol for q in NBR[lib])
                    reasons[('eye' if iseye else 'noeye', 'netcap' if cap_p>0 else 'nocap', 'la2' if la==2 else 'la3+')]+=1
                    if not iseye and cap_p==0 and la>=3 and shown<3:
                        print(gi,kk,g['net_color'],'chain',mem,'lib',lib,'la',la,'played',p,'top5',m['top5'])
                        print(pb(b)); shown+=1
                    break
print(reasons)
```