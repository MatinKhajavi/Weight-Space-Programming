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
for gi,k in ((1,88),(16,102),(21,82),(54,83),(59,53),(47,124)):
    rows,t5=show(gi,k)
    print(gi,k,G[gi]['net_color'],[(divmod(a,9) if a<81 else 'P',p) for a,p in t5])
    print('\n'.join(rows))
```