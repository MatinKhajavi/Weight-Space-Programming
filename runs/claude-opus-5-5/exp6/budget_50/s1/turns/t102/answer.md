```python
import json, numpy as np
exec(open('/work/goutil.py').read())
def regions(b):
    seen=set(); out=[]
    for p in range(81):
        if b[p]==0 and p not in seen:
            reg={p}; st=[p]; border=set()
            while st:
                x=st.pop()
                for y in NB[x]:
                    if b[y]==0 and y not in reg: reg.add(y); st.append(y)
                    elif b[y]!=0: border.add(b[y])
            seen|=reg; out.append((len(reg),border))
    return out
for dd in ('/work/pass_20/','/work/pass_21/'):
    G=json.load(open(dd+'games.json')); B=np.load(dd+'boards.npy')
    for g in G:
        if g['result']!=-1 or g['end']!='two passes' or g['margin']<-8: continue
        gi=g['game']; n=len(g['moves']); b=B[gi,n-1]; col=1 if g['net_color']=='black' else -1
        regs=regions(b)
        neu=sum(s for s,bd in regs if len(bd)!=1)
        own=sum(s for s,bd in regs if bd=={col}); opp=sum(s for s,bd in regs if bd=={-col})
        # weak chains (1 lib) of each color
        seen=set(); w1o=0; w1x=0
        for p in range(81):
            if b[p]!=0 and p not in seen:
                s,l=chain(b,p); seen|=s
                if len(l)==1:
                    if b[p]==col: w1o+=len(s)
                    else: w1x+=len(s)
        print(dd[-8:],gi,g['net_color'],g['margin'],'stones own/opp',int((b==col).sum()),int((b==-col).sum()),'terr own/opp',own,opp,'neutral',neu,'atari own/opp',w1o,w1x)
```