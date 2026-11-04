```python
exec(open('/work/ana.py').read()); exec(open('/work/go.py').read())
G,B,V=load_pass(2)
def trace(gi,last=8):
    g=G[gi]; mv=g['moves']; n=len(mv)
    for k in range(n-last,n):
        m=mv[k]
        prev=B[gi,k-1]; cur=B[gi,k]
        mvp=m['move']; who=m['by']
        print(k,who,divmod(mvp,9) if mvp<81 else 'pass', m.get('top5','')[:3] if who=='net' else '')
        rows=[]
        for r in range(9):
            s=''
            for c in range(9):
                v=int(cur[r*9+c]); ch={0:'.',1:'X',-1:'O'}[v]
                if r*9+c==mvp: ch=ch.lower() if ch!='.' else '*'
                s+=ch
            rows.append(s)
        print('\n'.join(rows))
trace(46,7)
```