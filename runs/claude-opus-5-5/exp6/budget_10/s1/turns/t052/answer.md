```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
G,boards=load_pass(5)
def regions(b):
    seen=set(); out=[]
    for p in range(81):
        if b[p]==0 and p not in seen:
            st=[p]; seen.add(p); reg=[]; cols=set()
            while st:
                x=st.pop(); reg.append(x)
                for y in NBR[x]:
                    if b[y]==0 and y not in seen: seen.add(y); st.append(y)
                    elif b[y]!=0: cols.add(int(b[y]))
            out.append((reg,cols))
    return out
for gi,g in enumerate(G):
    if g['opponent']!='strong' or g['result']>0 or g['margin']<-8: continue
    netcol = 1 if g['net_color']=='black' else -1
    n=len(g['moves']); b=boards[gi,n-1].astype(int)
    ch,idx=chains(b)
    neut=[(reg) for reg,cols in regions(b) if len(cols)==2]
    oppatari=[(len(mem),list(libs)) for col,mem,libs in ch if col==-netcol and len(libs)<=2]
    ownatari=[(len(mem),list(libs)) for col,mem,libs in ch if col==netcol and len(libs)<=2]
    print(gi,g['net_color'],g['margin'],'neutral regions',neut,'opp<=2libs',oppatari,'own<=2libs',ownatari)
```