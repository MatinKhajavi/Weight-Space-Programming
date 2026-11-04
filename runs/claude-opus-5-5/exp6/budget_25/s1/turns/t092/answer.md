```python
exec(open('/work/ana.py').read())
from collections import Counter
G,B,V=load_pass(19)
reps=[g for g in G if g['end']=='repetition']
def show(gi,k):
    m=G[gi]['moves'][k]; cur=B[gi,k-1]; mvp=m['move']
    rows=[]
    for r in range(9):
        s=''
        for c in range(9):
            ch={0:'.',1:'X',-1:'O'}[int(cur[r*9+c])]
            if r*9+c==mvp: ch='*'
            s+=ch
        rows.append(s)
    return rows, m.get('top5')
for g in reps:
    gi=g['game']; mv=g['moves']; n=len(mv); fb=B[gi,n-1]
    k0=[k for k in range(n-1) if (B[gi,k]==fb).all()][-1]
    seq=[(m['by'][0],divmod(m['move'],9) if m['move']<81 else 'P', round(m['top5'][0][1],2) if 'top5' in m else '') for m in mv[k0+1:n]]
    print(gi,g['result'],g['margin'],n-1-k0,seq)
g=[g for g in reps if g['result']<0][0]; gi=g['game']; mv=g['moves']; n=len(mv); fb=B[gi,n-1]
k0=[k for k in range(n-1) if (B[gi,k]==fb).all()][-1]
for k in range(k0+1,n):
    if mv[k]['by']=='net':
        rows,t5=show(gi,k); print(k,[(divmod(a,9) if a<81 else 'P',p) for a,p in t5][:3]); print('\n'.join(rows))
```