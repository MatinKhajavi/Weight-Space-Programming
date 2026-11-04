```python
exec(open('/work/ana.py').read())
G,B,V=load_pass(21)
for g in [g for g in G if g['end']=='repetition']:
    gi=g['game']; mv=g['moves']; n=len(mv); fb=B[gi,n-1]
    k0=[k for k in range(n-1) if (B[gi,k]==fb).all()][-1]
    seq=[(m['by'][0],divmod(m['move'],9) if m['move']<81 else 'P', round(m['top5'][0][1],2) if 'top5' in m else '') for m in mv[k0+1:n]]
    print(gi,g['result'],g['margin'],n-1-k0,seq)
```