```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
import collections
G,boards=load_pass(5)
st=collections.Counter(); lost=collections.Counter()
for gi,g in enumerate(G):
    if g['opponent']!='strong': continue
    netcol = 1 if g['net_color']=='black' else -1
    mv=g['moves']
    for kk,m in enumerate(mv):
        if m['by']!='opponent' or m['move']==81 or kk<2: continue
        b_before=boards[gi,kk-1].astype(int); b_after=boards[gi,kk].astype(int)
        capd=[p for p in range(81) if b_before[p]==netcol and b_after[p]==0]
        if not capd: continue
        n=len(capd)
        # state at our previous move (ply kk-1 was ours): board before our move = boards[kk-2]
        b0=boards[gi,kk-2].astype(int)
        ch,idx=chains(b0)
        # chains containing captured stones that existed at b0
        cids=set(idx[p] for p in capd if b0[p]==netcol)
        status=[]
        for c in cids:
            col,mem,libs=ch[c]
            if len(libs)==1:
                lib=list(libs)[0]; la,cap=libs_after(b0,lib,netcol)
                status.append('atari_sav' if (la and la>=2) else 'atari_unsav')
            elif len(libs)==2: status.append('2lib')
            else: status.append('3+lib')
        newstone = any(b0[p]==0 for p in capd)
        key=('ourmove_in_chain' if newstone else 'old')+'|'+','.join(sorted(status))
        st[key]+=1; lost[key]+=n
for k in sorted(st,key=lambda k:-lost[k]): print(k,st[k],lost[k])
```