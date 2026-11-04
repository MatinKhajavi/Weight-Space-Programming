```python
exec(open('/work/ana.py').read())
G,B,V=load_pass(2)
def trace(gi,last=8):
    g=G[gi]; mv=g['moves']; n=len(mv)
    out=[]
    for k in range(n-last,n):
        m=mv[k]; cur=B[gi,k]; mvp=m['move']
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
    for i in range(0,len(out),4):
        chunk=out[i:i+4]
        print('   '.join(h.ljust(12) for h,_ in chunk))
        for r in range(9): print('   '.join(rw[r].ljust(12) for _,rw in chunk))
trace(34,8)
```