```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
import collections
G,boards=load_pass(4)
st=collections.Counter(); sizes=collections.Counter()
for gi,g in enumerate(G):
    if g['opponent']!='strong': continue
    netcol = 1 if g['net_color']=='black' else -1
    for kk,m in enumerate(g['moves']):
        if m['by']!='net': continue
        b=(boards[gi,kk-1] if kk>0 else np.zeros(81)).astype(int)
        ch,idx=chains(b)
        own_at=[(mem,list(libs)[0]) for col,mem,libs in ch if col==netcol and len(libs)==1]
        if not own_at: continue
        for mem,lib in own_at:
            # savable by extension?
            la,cap=libs_after(b,lib,netcol)
            savable_ext = la is not None and la>=2
            # savable by capturing an adjacent opp chain in atari
            adjopp=set()
            for s in mem:
                for y in NBR[s]:
                    if b[y]==-netcol: adjopp.add(idx[y])
            cap_save=[list(ch[c][2])[0] for c in adjopp if len(ch[c][2])==1]
            sav = savable_ext or bool(cap_save)
            p=m['move']
            if p==81: act='pass'
            elif p==lib: act='extend'
            elif p in cap_save: act='capsave'
            else: act='other'
            st[(len(mem)>=3, sav, act)]+=1
for k in sorted(st,key=str): print(k,st[k])
```