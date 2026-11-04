```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
import collections
G,boards=load_pass(8)
# check self-atari and eye-fill moves overall
st=collections.Counter()
for gi,g in enumerate(G):
    if g['opponent']!='strong': continue
    netcol = 1 if g['net_color']=='black' else -1
    for kk,m in enumerate(g['moves']):
        if m['by']!='net' or m['move']==81: continue
        b=boards[gi,kk-1].astype(int); p=m['move']
        la,cap=libs_after(b,p,netcol)
        iseye=all(b[q]==netcol for q in NBR[p])
        if iseye: st['eyefill']+=1
        if la==1: st['SA_cap' if cap else 'SA_nocap']+=1
        st['moves']+=1
print(st)
for gi in [82,52]:
    g=G[gi]; n=len(g['moves'])
    print(gi,g['net_color'],g['margin']); 
    for k in [39,59,79,n-1]: print(pb(boards[gi,k]))
```