```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
import collections
G,boards=load_pass(6)
rows=sorted([(g['result'],g['net_color'][0],round(g['margin']),g['stones_captured_by_net'],g['stones_lost_by_net']) for g in G if g['opponent']=='strong'])
print(rows)
st=collections.Counter()
for gi,g in enumerate(G):
    if g['opponent']!='strong': continue
    netcol = 1 if g['net_color']=='black' else -1
    for kk,m in enumerate(g['moves']):
        if m['by']!='net' or m['move']==81: continue
        b=boards[gi,kk-1].astype(int); p=m['move']
        la,cap=libs_after(b,p,netcol)
        if la==1: st['SA_cap' if cap else 'SA_nocap']+=1
        st['moves']+=1
print(st)
```