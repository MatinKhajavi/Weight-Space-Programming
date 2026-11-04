```python
import numpy as np, json
exec(open('/work/go.py').read())
d='/work/pass_1/'
A5=np.load(d+'act_L5.npy'); A6=np.load(d+'act_L6.npy'); X=np.load(d+'inputs.npy'); LG=np.load(d+'logits.npy')
req=json.load(open('/work/test1_req.json')); poss=req['positions'][:30]
stats={'atari_fn':0,'atari_fp':0,'lt3_fn':0,'lt3_fp':0,'n':0}
fpex=[]
for i,ps in enumerate(poss):
    bd=[{'.':0,'X':1,'O':2}[ch] for row in ps['board'] for ch in row]
    me=1 if ps['to_move']=='black' else 2
    lc=libcounts(bd)
    # check inputs
    for p in range(81):
        if bd[p]==0: continue
        own = bd[p]==me
        a = A5[i,(164 if own else 245)+p]; l=A5[i,(326 if own else 407)+p]
        stats['n']+=1
        ta = lc[p]==1; tl = lc[p]<=2
        if ta and a<0.5: stats['atari_fn']+=1
        if (not ta) and a>0.5: stats['atari_fp']+=1; fpex.append((i,p,lc[p],len(group(bd,p)[0])))
        if tl and l<0.5: stats['lt3_fn']+=1
        if (not tl) and l>0.5: stats['lt3_fp']+=1
print(stats)
print(fpex[:20])
print(np.unique(A5[:, 164:488]))
```