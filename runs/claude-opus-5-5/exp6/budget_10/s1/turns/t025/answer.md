```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
import collections
def tstats(k):
    G,boards=load_pass(k)
    st=collections.Counter()
    for gi,g in enumerate(G):
        mv=g['moves']; netcol = 1 if g['net_color']=='black' else -1
        prev_cap_single=None
        for kk,m in enumerate(mv):
            if m['by']!='net' or m['move']==81: continue
            b=(boards[gi,kk-1] if kk>0 else np.zeros(81)).astype(int)
            p=m['move']; la,cap=libs_after(b,p,netcol)
            ch,idx=chains(b)
            caps=[(len(mem),list(libs)[0]) for col,mem,libs in ch if col==-netcol and len(libs)==1]
            legal=[c for c in caps if play(b,c[1],netcol)[0] is not None]
            st['moves']+=1
            if legal:
                st['cap_avail']+=1
                big=max(c[0] for c in legal)
                if cap>0: st['captured']+=1
                elif big>=2: st['missed_big']+=1
            if la==1 and cap==0: st['SA_nocap']+=1
            # filled a liberty of opp chain with >=2 libs while being SA
            own_at=[1 for col,mem,libs in ch if col==netcol and len(libs)==1]
            if own_at: st['own_atari']+=1
    return st
print(2,tstats(2)); print(3,tstats(3))
```