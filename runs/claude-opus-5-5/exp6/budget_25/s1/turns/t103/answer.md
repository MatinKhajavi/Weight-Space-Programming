```python
exec(open('/work/ana.py').read())
G,B,V=load_pass(21)
def grid(gi,plies):
    mv=G[gi]['moves']; out=[]
    for k in plies:
        m=mv[k]; cur=B[gi,k]; mvp=m['move']
        hdr=f"{k}{m['by'][0]}{divmod(mvp,9) if mvp<81 else 'P'}"
        rows=[]
        for r in range(9):
            s=''
            for c in range(9):
                ch={0:'.',1:'X',-1:'O'}[int(cur[r*9+c])]
                if r*9+c==mvp: ch=ch.lower() if ch!='.' else '*'
                s+=ch
            rows.append(s)
        out.append((hdr,rows))
    print(' '.join(h.ljust(9) for h,_ in out))
    for r in range(9): print(' '.join(rw[r].ljust(9) for _,rw in out))
g=G[23]; n=len(g['moves']); print(g['net_color'])
grid(23,list(range(n-6,n)))
```