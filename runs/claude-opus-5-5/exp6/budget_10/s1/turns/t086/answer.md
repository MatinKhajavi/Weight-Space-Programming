```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
import collections
G,boards=load_pass(9)
st=collections.Counter(); ex=[]
for gi,g in enumerate(G):
    if g['opponent']!='strong': continue
    netcol = 1 if g['net_color']=='black' else -1
    for kk,m in enumerate(g['moves']):
        if m['by']!='net' or m['move']==81: continue
        b=boards[gi,kk-1].astype(int); p=m['move']
        la,cap=libs_after(b,p,netcol)
        if la!=1 or cap>0: continue
        nb,_=play(b,p,netcol); ch2,idx2=chains(nb); sz=len(ch2[idx2[p]][1])
        # was the move's chain going to be captured anyway? sizes of own chains adjacent before
        ch,idx=chains(b)
        adj=[(len(ch[idx[q]][1]),len(ch[idx[q]][2])) for q in NBR[p] if b[q]==netcol]
        iseye=all(b[q]==netcol for q in NBR[p])
        st[('eye' if iseye else 'noeye', 'big' if sz>=5 else 'small')]+=1
        if sz>=5: ex.append((gi,kk,p,sz,adj,g['result'],m['top5'][:2]))
print(st); 
for e in ex: print(e)
```