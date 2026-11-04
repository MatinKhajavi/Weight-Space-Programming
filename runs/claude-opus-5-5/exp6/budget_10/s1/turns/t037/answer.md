```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
import collections
G,boards=load_pass(4)
cnt=collections.Counter()
for gi,g in enumerate(G):
    if g['end']=='repetition':
        cnt[(g['opponent'],g['moves'][-1]['by'],g['result'])]+=1
print(cnt)
for gi,g in enumerate(G):
    if g['end']!='repetition' or g['result']>0: continue
    netcol = 1 if g['net_color']=='black' else -1
    mv=g['moves']; n=len(mv); out=[]
    for kk in range(max(0,n-8),n):
        m=mv[kk]
        if m['by']!='net' or m['move']==81: continue
        b=boards[gi,kk-1].astype(int); p=m['move']
        la,cap=libs_after(b,p,netcol)
        ch,idx=chains(b)
        adj_opp=[(len(ch[idx[q]][1]),len(ch[idx[q]][2])) for q in NBR[p] if b[q]==-netcol]
        adj_own=[(len(ch[idx[q]][1]),len(ch[idx[q]][2])) for q in NBR[p] if b[q]==netcol]
        out.append((kk,p,'la',la,'cap',cap,'opp(sz,L)',adj_opp,'own',adj_own,m['top5'][0][1]))
    print(gi,g['opponent'],g['margin'],out[-3:])
```