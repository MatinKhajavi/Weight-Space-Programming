```python
exec(open('/work/ana.py').read())
G,B,V=load_pass(20)
gl=sorted([g for g in G if g['result']<0],key=lambda g:g['margin'])[0]
gi=gl['game']; mv=gl['moves']; print(gi,gl['net_color'],gl['margin'])
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
me=1 if gl['net_color']=='black' else -1
caps=[(k,int(((B[gi,k-1]==me)&(B[gi,k]==0)).sum())) for k in range(1,len(mv))]
caps=[c for c in caps if c[1]>=2]; print(caps)
k0=caps[0][0]
grid(gi,list(range(max(0,k0-13),k0+1)))
for k in range(max(0,k0-13),k0+1):
    if mv[k]['by']=='net': print(k,[(divmod(a,9) if a<81 else 'P',round(p,2)) for a,p in mv[k]['top5'][:3]])
```