```python
exec(open('/work/ana.py').read())
G,B,V=load_pass(2)
def grid(gi,plies):
    g=G[gi]; mv=g['moves']
    out=[]
    for k in plies:
        m=mv[k]; cur=B[gi,k]; mvp=m['move']
        t5=m.get('top5')
        hdr=f"{k}{m['by'][0]}{divmod(mvp,9) if mvp<81 else 'P'}"
        rows=[]
        for r in range(9):
            s=''
            for c in range(9):
                ch={0:'.',1:'X',-1:'O'}[int(cur[r*9+c])]
                if r*9+c==mvp: ch=ch.lower()
                s+=ch
            rows.append(s)
        out.append((hdr,rows))
    for i in range(0,len(out),6):
        chunk=out[i:i+6]
        print('  '.join(h.ljust(10) for h,_ in chunk))
        for r in range(9): print('  '.join(rw[r].ljust(10) for _,rw in chunk))
g=G[36]
print([(m['ply'],m['move'],m['top5'][0][1] if 'top5' in m else '') for m in g['moves'][:40]])
grid(36,list(range(1,48,2)))
```