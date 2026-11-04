```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
import collections
def diags(p):
    r,c=divmod(p,9); o=[]
    for dr in (-1,1):
        for dc in (-1,1):
            rr,cc=r+dr,c+dc
            if 0<=rr<9 and 0<=cc<9: o.append(rr*9+cc)
    return o
st=collections.Counter()
for k in (8,9):
    G,boards=load_pass(k)
    for gi,g in enumerate(G):
        netcol = 1 if g['net_color']=='black' else -1
        for kk,m in enumerate(g['moves']):
            if m['by']!='net': continue
            b=boards[gi,kk-1].astype(int)
            # consider all eye points with an adjacent own chain in atari (candidate escapes)
            ch,idx=chains(b)
            for p in range(81):
                if b[p]!=0 or not all(b[q]==netcol for q in NBR[p]): continue
                if not any(len(ch[idx[q]][2])==1 for q in NBR[p]): continue
                la,cap=libs_after(b,p,netcol)
                if la is None: continue
                # local gating: isolated atari stone adjacent (all other nbrs opp)
                gate=any(all(b[t]==-netcol for t in NBR[q] if t!=p) for q in NBR[p])
                od=sum(1 for d in diags(p) if b[d]==netcol)
                st[('gate' if gate else 'nogate','la1' if la==1 else 'la2+','owndiag=%d'%od)]+=1
for kk in sorted(st,key=str): print(kk,st[kk])
```