```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
import collections
G,boards=load_pass(5)
st=collections.Counter()
for gi,g in enumerate(G):
    netcol = 1 if g['net_color']=='black' else -1
    for kk,m in enumerate(g['moves']):
        if m['by']!='net' or kk%3: continue
        b=boards[gi,kk-1].astype(int)
        ch,idx=chains(b)
        for p in range(81):
            if b[p]!=0: continue
            la,cap=libs_after(b,p,netcol)
            if la is None: continue
            e=sum(1 for q in NBR[p] if b[q]==0)
            v=collections.Counter(); seen=set()
            capc=0
            for q in NBR[p]:
                if b[q]==netcol:
                    L=len(ch[idx[q]][2]); sz=len(ch[idx[q]][1]); lq=sum(1 for y in NBR[q] if b[y]==0)
                    s1 = 1 if L>=2 else 0
                    v['A']+=s1
                    v['loc']+=s1 + (1 if (L>=2 and (lq>=3 or sz>=2)) else 0)
                    v['l3']+=s1 + (1 if (L>=2 and lq>=3) else 0)
                    if idx[q] not in seen:
                        seen.add(idx[q]); v['chainexact']+=min(2,L-1)
                elif b[q]==-netcol and len(ch[idx[q]][2])==1 and len(ch[idx[q]][1])==1: capc+=1
            trueSA=(la==1)
            for name in ('A','loc','l3','chainexact'):
                est = e+v[name]+2*capc <= 1
                st[(name,'trueSA' if trueSA else 'ok','est' if est else 'no')]+=1
for k in sorted(st,key=str): print(k,st[k])
```