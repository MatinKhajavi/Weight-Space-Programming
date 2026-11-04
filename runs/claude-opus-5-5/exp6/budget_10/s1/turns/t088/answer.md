```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
import collections
st=collections.Counter()
for k_ in (8,9):
    G,boards=load_pass(k_)
    for gi,g in enumerate(G):
        netcol = 1 if g['net_color']=='black' else -1
        for kk,m in enumerate(g['moves']):
            if m['by']!='net': continue
            b=boards[gi,kk-1].astype(int)
            ch,idx=chains(b)
            for p in range(81):
                if b[p]!=0 or not all(b[q]==netcol for q in NBR[p]): continue
                if not any(len(ch[idx[q]][2])==1 for q in NBR[p]): continue
                la,cap=libs_after(b,p,netcol)
                if la is None: continue
                gate=any(all(b[t]==-netcol for t in NBR[q] if t!=p) for q in NBR[p])
                if not gate: continue
                others=[q for q in NBR[p] if len(ch[idx[q]][2])>=2]
                nchains=len(set(idx[q] for q in others))
                A=0
                for q in others:
                    lq=sum(1 for y in NBR[q] if b[y]==0); A+=1+max(0,lq-2)
                st[('la1' if la==1 else 'la2+', 'k=%d'%len(others), 'nch=%d'%nchains, 'A<=1' if A<=1 else 'A>=2')]+=1
for kk in sorted(st,key=str): print(kk,st[kk])
```