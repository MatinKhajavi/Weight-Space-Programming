```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_32/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
from collections import Counter
c=Counter((g['end'],g['result'],g['moves'][-1]['by']) for g in G)
print(c)
def regions_score(b,col):
    seen=set(); own=0; opp=0
    for p in range(81):
        if b[p]==0 and p not in seen:
            reg={p}; st=[p]; border=set()
            while st:
                x=st.pop()
                for y in NB[x]:
                    if b[y]==0 and y not in reg: reg.add(y); st.append(y)
                    elif b[y]!=0: border.add(b[y])
            seen|=reg
            if border=={col}: own+=len(reg)
            elif border=={-col}: opp+=len(reg)
    return own,opp
for g in G:
    if g['result']==-1 and g['end']=='repetition':
        gi=g['game']; mv=g['moves']; n=len(mv); col=1 if g['net_color']=='black' else -1
        last=mv[-1]; prev=B[gi,n-2]
        nb,cap=play(prev,last['move'],col) if last['move']<81 else (prev,0)
        ne=sum(1 for y in NB[last['move']] if prev[y]==0) if last['move']<81 else -1
        lead=int((prev==col).sum()-(prev==-col).sum())+(7.5 if col==-1 else -7.5)
        to,tp=regions_score(prev,col)
        print(gi,g['net_color'],g['margin'],'stone lead',lead,'terr',to,tp,'cap',cap,'ne',ne,last['top5'][:2])
```