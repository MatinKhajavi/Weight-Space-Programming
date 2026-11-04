```python
exec(open('/work/ana.py').read())
from collections import Counter
G,B,V=load_pass(5)
print(Counter((g['opponent'],g['end'],g['result']) for g in G))
reps=[g for g in G if g['end']=='repetition']
for g in reps[:14]:
    gi=g['game']; mv=g['moves']; n=len(mv)
    fb=B[gi,n-1]
    prev=[k for k in range(n-1) if (B[gi,k]==fb).all()]
    k0=prev[-1]
    seq=[(m['by'][0],divmod(m['move'],9) if m['move']<81 else 'P', m['top5'][0][1] if 'top5' in m else '') for m in mv[k0+1:n]]
    print(gi,g['opponent'],g['net_color'],g['result'],g['margin'],'from',k0,'len',n-1-k0, seq)
```