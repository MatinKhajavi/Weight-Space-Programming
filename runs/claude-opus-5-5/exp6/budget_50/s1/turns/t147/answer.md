```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_37/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def neutral_info(b,col):
    seen=set(); out=[]
    for p in range(81):
        if b[p]==0 and p not in seen:
            reg={p}; st=[p]; border=set()
            while st:
                x=st.pop()
                for y in NB[x]:
                    if b[y]==0 and y not in reg: reg.add(y); st.append(y)
                    elif b[y]!=0: border.add(b[y])
            seen|=reg
            if len(border)==2: out.append(len(reg))
    return out
tot=0
for g in G:
    if g['result']!=-1 or g['end']!='two passes': continue
    gi=g['game']; n=len(g['moves']); col=1 if g['net_color']=='black' else -1; b=B[gi,n-1]
    neu=neutral_info(b,col)
    # legal non-self-atari moves for us at the end that touch neutral regions
    good=[]
    for p in range(81):
        if b[p]!=0: continue
        nb,cap=play(b,p,col)
        if nb is None: continue
        s,l=chain(nb,p)
        if len(l)>=2 and any(b[y]==-col for y in NB[p]): good.append(divmod(p,9))
    print(gi,g['net_color'],g['margin'],'neutral regions',neu,'safe contact moves',good[:6])
```