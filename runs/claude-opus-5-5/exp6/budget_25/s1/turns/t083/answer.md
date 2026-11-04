```python
exec(open('/work/ana.py').read())
G,B,V=load_pass(16)
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
for gi in (4,87,92):
    g=G[gi]; mv=g['moves']; n=len(mv)
    rows,t5=show(gi,n-3)
    print(gi,g['net_color'],[(divmod(a,9) if a<81 else 'P',p) for a,p in t5][:3])
    print('\n'.join(rows))
```