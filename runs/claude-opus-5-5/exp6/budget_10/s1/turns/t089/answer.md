```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
import collections
st=collections.Counter()
for k_ in (8,9):
    G,boards=load_pass(k_)
    for gi,g in enumerate(G):
        netcol = 1 if g['net_color']=='black' else -1
        for kk,m in enumerate(g['moves']):
            if m['by']!='net' or m['move']==81: continue
            b=boards[gi,kk-1].astype(int); p=m['move']
            if not all(b[q]==netcol for q in NBR[p]): continue
            ch,idx=chains(b)
            if not any(len(ch[idx[q]][2])==1 for q in NBR[p]): continue
            la,cap=libs_after(b,p,netcol)
            deg=len(NBR[p])
            # survival: is p still our stone 10 plies later?
            n=len(g['moves']); later=boards[gi,min(n-1,kk+10)]
            surv = later[p]==netcol
            st[('deg%d'%deg,'la1' if la==1 else 'la2+','surv' if surv else 'died')]+=1
for kk in sorted(st,key=str): print(kk,st[kk])
```