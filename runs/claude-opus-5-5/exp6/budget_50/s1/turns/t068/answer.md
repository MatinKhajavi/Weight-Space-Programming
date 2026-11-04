```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_14/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def neutral_list(b):
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
            if len(border)!=1: out+=sorted(reg)
    return out
for gi in (6,10,19,28,56,66,89,94,111,114):
    g=G[gi]; col=1 if g['net_color']=='black' else -1
    fp=[m for m in g['moves'] if m['move']==81 and m['by']=='net'][0]; k=fp['ply']; b=B[gi,k-1]
    neu=neutral_list(b)
    info=[]
    for p in neu:
        nb,cap=play(b,p,col)
        if nb is None: info.append((divmod(p,9),'illegal')); continue
        s,l=chain(nb,p)
        info.append((divmod(p,9),'libs',len(l),'cap',cap))
    print(gi,g['net_color'],g['margin'],info)
gi=6; g=G[gi]; fp=[m for m in g['moves'] if m['move']==81 and m['by']=='net'][0]; k=fp['ply']
for s in to_strings(B[gi,k-1]): print('   ',s)
```