```python
exec(open('/work/ana.py').read())
from collections import Counter
G,B,V=load_pass(16)
reps=[g for g in G if g['end']=='repetition']
c=Counter()
for g in reps:
    gi=g['game']; mv=g['moves']; n=len(mv); fb=B[gi,n-1]
    k0=[k for k in range(n-1) if (B[gi,k]==fb).all()][-1]
    seq=[(m['by'][0],m['move']) for m in mv[k0+1:n]]
    c[(n-1-k0, g['result'])]+=1
    print(gi,g['result'],g['margin'],n-1-k0,[(b,divmod(x,9) if x<81 else 'P') for b,x in seq])
print(c)
```