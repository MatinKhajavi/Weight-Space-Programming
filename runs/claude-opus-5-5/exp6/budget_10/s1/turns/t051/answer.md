```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
G,boards=load_pass(5)
k=0
for gi,g in enumerate(G):
    if g['opponent']!='strong' or g['result']>0 or g['margin']<-6: continue
    n=len(g['moves']); b=boards[gi,n-1].astype(int)
    print(gi,g['net_color'],g['margin'],'last moves',[(m['by'][0],m['move']) for m in g['moves'][-6:]])
    print(pb(b)); k+=1
    if k>=5: break
```