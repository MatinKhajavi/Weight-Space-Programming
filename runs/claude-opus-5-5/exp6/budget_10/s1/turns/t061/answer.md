```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
G,boards=load_pass(6)
def area_margin(b,netcol):
    seen=set(); s={1:int((b==1).sum()),-1:int((b==-1).sum())}
    for p in range(81):
        if b[p]==0 and p not in seen:
            st=[p]; seen.add(p); reg=[]; cols=set()
            while st:
                x=st.pop(); reg.append(x)
                for y in NBR[x]:
                    if b[y]==0 and y not in seen: seen.add(y); st.append(y)
                    elif b[y]!=0: cols.add(int(b[y]))
            if len(cols)==1: s[cols.pop()]+=len(reg)
    m=s[1]-s[-1]-7.5
    return m if netcol==1 else -m
for gi,g in enumerate(G):
    if g['opponent']!='strong' or g['result']>0 or g['margin']<-6: continue
    netcol = 1 if g['net_color']=='black' else -1
    mv=g['moves']; first=None
    for kk,m in enumerate(mv):
        if m['by']=='net' and m['move']==81: first=kk; break
    if first is None: continue
    marg_at=area_margin(boards[gi,first-1].astype(int),netcol)
    after=[(m['by'][0],m['move']) for m in mv[first:]]
    opp_moves_after=sum(1 for m in mv[first:] if m['by']=='opponent' and m['move']!=81)
    print(gi,g['net_color'],'final',g['margin'],'first pass ply',first,'of',len(mv),'margin then',marg_at,'opp moves after',opp_moves_after, after[:16])
```