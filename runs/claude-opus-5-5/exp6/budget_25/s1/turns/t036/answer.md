```python
exec(open('/work/ana.py').read())
G,B,V=load_pass(5)
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
for gi in (96,97,109,124,126):
    g=G[gi]; mv=g['moves']; n=len(mv); fb=B[gi,n-1]
    k0=[k for k in range(n-1) if (B[gi,k]==fb).all()][-1]
    seq=[(m['by'][0],divmod(m['move'],9) if m['move']<81 else 'P') for m in mv[k0+1:n]]
    k=k0+1
    rows,t5=show(gi,k)
    print(gi,seq,[(divmod(a,9) if a<81 else 'P',p) for a,p in (t5 or [])][:3])
    print('\n'.join(rows))
```