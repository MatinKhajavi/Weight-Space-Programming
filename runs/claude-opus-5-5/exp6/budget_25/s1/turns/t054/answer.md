```python
exec(open('/work/ana.py').read())
G,B,V=load_pass(9)
def grid(gi,plies):
    g=G[gi]; mv=g['moves']; out=[]
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
    for i in range(0,len(out),7):
        chunk=out[i:i+7]
        print(' '.join(h.ljust(9) for h,_ in chunk))
        for r in range(9): print(' '.join(rw[r].ljust(9) for _,rw in chunk))
ls=[g for g in G if g['opponent']=='strong' and g['result']<0 and g['end']=='two passes']
g=ls[3]; gi=g['game']; print(gi,g['net_color'],g['margin'],len(g['moves']))
grid(gi,list(range(10,len(g['moves']),6)))
```