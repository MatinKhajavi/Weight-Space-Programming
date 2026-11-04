```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
G,boards=load_pass(6)
gi=52; g=G[gi]; mv=g['moves']
first=[kk for kk,m in enumerate(mv) if m['by']=='net' and m['move']==81][0]
print(g['net_color'])
for kk in range(first-1, min(first+12,len(mv))):
    m=mv[kk]; print(kk,m['by'],divmod(m['move'],9) if m['move']<81 else 'PASS', m.get('top5',[])[:3])
    print(pb(boards[gi,kk]))
```