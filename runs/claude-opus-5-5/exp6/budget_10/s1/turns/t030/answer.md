```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
G,boards=load_pass(3)
for gi,g in enumerate(G):
    if g['end']!='repetition' or g['result']>0: continue
    netcol = 1 if g['net_color']=='black' else -1
    mv=g['moves']; n=len(mv)
    # find SA moves by net in last 12 plies
    out=[]
    for kk in range(max(0,n-12),n):
        m=mv[kk]
        if m['by']!='net' or m['move']==81: continue
        b=boards[gi,kk-1].astype(int); p=m['move']
        la,cap=libs_after(b,p,netcol)
        ch,idx=chains(b)
        adj_opp=[len(ch[idx[q]][2]) for q in NBR[p] if b[q]==-netcol]
        adj_own=[len(ch[idx[q]][2]) for q in NBR[p] if b[q]==netcol]
        out.append((kk,p,'libs_after',la,'cap',cap,'oppL',adj_opp,'ownL',adj_own))
    print(gi,g['opponent'],out[-3:])
```