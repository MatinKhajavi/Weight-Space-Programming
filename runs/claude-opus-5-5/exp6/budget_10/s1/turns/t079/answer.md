```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
import collections
G,boards=load_pass(8)
st=collections.Counter(); ex=[]
for gi,g in enumerate(G):
    if g['opponent']!='strong': continue
    netcol = 1 if g['net_color']=='black' else -1
    for kk,m in enumerate(g['moves']):
        if m['by']!='net' or m['move']==81: continue
        b=boards[gi,kk-1].astype(int); p=m['move']
        if not all(b[q]==netcol for q in NBR[p]): continue
        ch,idx=chains(b)
        atari=any(len(ch[idx[q]][2])==1 for q in NBR[p])
        la,cap=libs_after(b,p,netcol)
        # how many other eyes (single-point eyes) does merged group have
        nb,_=play(b,p,netcol); ch2,idx2=chains(nb); grp=ch2[idx2[p]]
        libs=grp[2]
        eyes=sum(1 for l in libs if all(nb[y]==netcol for y in NBR[l]))
        key=('atari' if atari else 'noatari', 'la1' if la==1 else ('la2' if la==2 else 'la3+'), 'eyes_left=%d'%min(eyes,2))
        st[key]+=1
        if not atari and len(ex)<4 and eyes<2: ex.append((gi,kk,p,m['top5'][:3]))
for k in sorted(st,key=str): print(k,st[k])
print(ex)
```