```python
exec(open('/work/ana.py').read())
G,B,V=load_pass(17)
gi=5; g=G[gi]; mv=g['moves']
# find plies where net lost >=5 stones at once
caps=[]
for k in range(1,len(mv)):
    me=1 if g['net_color']=='black' else -1
    lost=int(((B[gi,k-1]==me)&(B[gi,k]==0)).sum())
    if lost>=3: caps.append((k,lost))
print(caps)
def grid(gi,plies):
    out=[]
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
k0=caps[0][0]
grid(gi,list(range(k0-12,k0+1)))
for k in range(k0-12,k0+1):
    if mv[k]['by']=='net': print(k,[(divmod(a,9) if a<81 else 'P',p) for a,p in mv[k]['top5'][:3]])
```