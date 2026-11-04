```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
import collections
R2=[[q for q in range(81) if 0<abs(q//9-p//9)+abs(q%9-p%9)<=2] for p in range(81)]
R3=[[q for q in range(81) if 0<abs(q//9-p//9)+abs(q%9-p%9)<=3] for p in range(81)]
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
            e2=sum(1 for r in R2[p] if b[r]==0); e3=sum(1 for r in R3[p] if b[r]==0)
            st[('la1' if la==1 else 'la2+', 'e2=%d'%min(e2,4))]+=1
            st[('la1' if la==1 else 'la2+', 'e3=%d'%min(e3,6))]+=1
for kk in sorted(st,key=str): print(kk,st[kk])
```