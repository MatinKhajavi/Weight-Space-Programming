```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_2/'
G=json.load(open(d+'games.json'))
B=np.load(d+'boards.npy')
false_caps=[]; true_caps=0; tot=0
for g in G:
    gi=g['game']; col=1 if g['net_color']=='black' else -1
    for m in g['moves']:
        if m['by']!='net' or m['move']==81: continue
        k=m['ply']; prev=B[gi,k-1] if k>0 else np.zeros(81,int)
        p=m['move']
        # does it capture?
        nb,cap=play(prev,p,col)
        # is any adjacent opp chain in atari (true)?
        adj_atari=any(prev[y]==-col and len(chain(prev,y)[1])==1 for y in NB[p])
        top=m['top5'][0]
        if cap>0: true_caps+=1
        tot+=1
        if top[1]>0.9 and top[0]==p and cap==0:
            # what kind? own atari adjacent?
            own_atari=any(prev[y]==col and len(chain(prev,y)[1])==1 for y in NB[p])
            if not own_atari:
                # opp chains adjacent: sizes and libs
                info=[(len(chain(prev,y)[0]),len(chain(prev,y)[1])) for y in NB[p] if prev[y]==-col]
                false_caps.append((g['opponent'],gi,k,divmod(p,9),info))
print('net moves',tot,'captures',true_caps,'confident non-capture non-escape',len(false_caps))
for x in false_caps[:40]: print(x)
```