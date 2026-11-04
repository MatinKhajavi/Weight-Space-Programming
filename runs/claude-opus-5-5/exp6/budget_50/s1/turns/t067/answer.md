```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_14/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
def neutral_points(b):
    # empty points not belonging to a single-color region
    seen=set(); neu=0
    for p in range(81):
        if b[p]==0 and p not in seen:
            reg={p}; st=[p]; border=set()
            while st:
                x=st.pop()
                for y in NB[x]:
                    if b[y]==0 and y not in reg: reg.add(y); st.append(y)
                    elif b[y]!=0: border.add(b[y])
            seen|=reg
            if len(border)!=1: neu+=len(reg)
    return neu
for g in G:
    if g['result']!=-1 or g['end']!='two passes': continue
    gi=g['game']; mv=g['moves']
    ps=[(m['ply'],m['by'][0]) for m in mv if m['move']==81]
    fp=[m for m in mv if m['move']==81 and m['by']=='net']
    if not fp: continue
    k=fp[0]['ply']; b=B[gi,k-1]
    top=fp[0]['top5'][:3]
    print(gi,g['net_color'],g['margin'],'first net pass ply',k,'neutral empties at that time',neutral_points(b),'passes',ps[:6],'top',top)
```