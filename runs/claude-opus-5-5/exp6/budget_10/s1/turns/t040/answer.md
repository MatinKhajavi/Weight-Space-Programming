```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
G,boards=load_pass(4)
shown=0
for gi,g in enumerate(G):
    if g['opponent']!='strong': continue
    netcol = 1 if g['net_color']=='black' else -1
    for kk,m in enumerate(g['moves']):
        if m['by']!='net' or m['move']!=81: continue
        b=boards[gi,kk-1].astype(int)
        ch,idx=chains(b)
        for col,mem,libs in ch:
            if col==netcol and len(libs)==1:
                lib=list(libs)[0]; la,cap=libs_after(b,lib,netcol)
                if la and la>=2 and len(mem)<3:
                    print(gi,kk,g['net_color'],'chain',mem,'lib',lib,'la',la,'top5',m['top5'])
                    print(pb(b)); shown+=1
                    break
        if shown>=3: break
    if shown>=3: break
```