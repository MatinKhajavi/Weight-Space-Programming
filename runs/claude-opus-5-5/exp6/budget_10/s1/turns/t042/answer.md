```python
exec(open('/work/go.py').read()); exec(open('/work/an.py').read())
import collections
G,boards=load_pass(4)
valid=np.load('/work/pass_4/valid.npy')
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
            sup=0; sup2=0; capc=0; atari_adj=False
            for q in NBR[p]:
                if b[q]==netcol:
                    L=len(ch[idx[q]][2]); sz=len(ch[idx[q]][1])
                    if L>=2: sup+=1
                    if L==1: atari_adj=True
                    lq=sum(1 for y in NBR[q] if b[y]==0)
                    if L>=2 and (lq>=3 or sz>=2): sup2+=1
                elif b[q]==-netcol and len(ch[idx[q]][2])==1 and len(ch[idx[q]][1])==1: capc+=1
            A=e+sup+2*capc; A2=e+sup+sup2+2*capc
            trueSA = (la==1)
            for name,AA in (('A',A),('A+loc',A2)):
                est=AA<=1
                st[(name,'adjAtari' if atari_adj else 'other', 'trueSA' if trueSA else 'ok', 'est' if est else 'no')]+=1
for k in sorted(st,key=str): print(k,st[k])
```