```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
import collections
G,boards=load_pass(8)
res=collections.defaultdict(lambda: [0.0,0,0.0,0])  # sum pen on trueSA, count, sum pen on ok, count
for gi,g in enumerate(G):
    netcol = 1 if g['net_color']=='black' else -1
    for kk,m in enumerate(g['moves']):
        if m['by']!='net' or kk%4: continue
        b=boards[gi,kk-1].astype(int)
        ch,idx=chains(b)
        for p in range(81):
            if b[p]!=0: continue
            la,cap=libs_after(b,p,netcol)
            if la is None: continue
            e=sum(1 for q in NBR[p] if b[q]==0); nown=0; capc=0
            sup=collections.Counter()
            for q in NBR[p]:
                if b[q]==netcol:
                    nown+=1
                    L=len(ch[idx[q]][2]); lq=sum(1 for y in NBR[q] if b[y]==0)
                    safe=1 if L>=2 else 0; ext=max(0,lq-2)
                    for a in (1.0,0.75,0.5):
                        sup[a]+=a*safe+ext
                elif b[q]==-netcol and len(ch[idx[q]][2])==1 and len(ch[idx[q]][1])==1: capc+=1
            for a in (1.0,0.75,0.5):
                A=e+sup[a]+2*capc
                pen=max(0,2-A) - 2*max(0,1-A-nown)
                r=res[a]
                if la==1 and cap==0: r[0]+=min(pen,1); r[1]+=1
                elif la>=2 or cap>0: r[2]+=min(pen,1); r[3]+=1
for a,r in res.items(): print(a,'mean pen on trueSA %.3f (n=%d)  on OK %.4f (n=%d)  missedSA-mass %.1f falseSA-mass %.1f'%(r[0]/r[1],r[1],r[2]/r[3],r[3], r[1]-r[0], r[2]))
```