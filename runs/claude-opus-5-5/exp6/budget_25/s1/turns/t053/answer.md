```python
exec(open('/work/ana.py').read())
G,B,V=load_pass(9)
ws=[g for g in G if g['opponent']=='weak']
ws.sort(key=lambda g:g['margin'])
for g in ws[:3]:
    gi=g['game']; mv=g['moves']; n=len(mv)
    fp=[m['ply'] for m in mv if m['by']=='net' and m['move']==81]
    print(gi,g['net_color'],g['margin'],g['result'],'passes',fp,'len',n)
    k=fp[0]
    print(sb(B[gi,k-1]))
    print([(divmod(a,9) if a<81 else 'P',p) for a,p in mv[k]['top5']])
    print(sb(B[gi,n-1]))
```