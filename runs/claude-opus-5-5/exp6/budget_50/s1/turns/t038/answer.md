```python
import json, numpy as np
exec(open('/work/goutil.py').read())
d='/work/pass_7/'
G=json.load(open(d+'games.json')); B=np.load(d+'boards.npy')
falsepos=[]; truepos=[]
for g in G:
    if g['opponent']!='strong': continue
    gi=g['game']; col=1 if g['net_color']=='black' else -1
    mvs=[m['move'] for m in g['moves']]
    for m in g['moves']:
        if m['by']!='net' or m['move']==81: continue
        k=m['ply']; prev=B[gi,k-1]; p=m['move']
        nb,cap=play(prev,p,col)
        hist=[x if x<81 else "pass" for x in mvs[:k]]
        if cap==0 and m['top5'][0][0]==p and m['top5'][0][1]>0.9:
            opp=[y for y in NB[p] if prev[y]==-col]
            if opp and len(chain(prev,opp[0])[0])>=10:
                falsepos.append((gi,k,p,hist))
        if cap>=5:
            truepos.append((gi,k,p,cap,hist))
print(len(falsepos),len(truepos))
sel=falsepos[::max(1,len(falsepos)//12)][:12]+truepos[::max(1,len(truepos)//8)][:8]
meta=[(x[0],x[1],x[2],x[3] if len(x)==5 else 0) for x in sel]
json.dump(meta,open('/work/testpos2_meta.json','w'))
pos=[{"moves":x[-1]} for x in sel]
json.dump(pos,open('/work/testpos2.json','w'))
s=json.dumps({"positions":pos,"activations":False})
print(len(s)); print(s)
```