```python
import numpy as np, json
exec(open('/work/go.py').read())
d='/work/pass_12/'
A5=np.load(d+'act_L5.npy'); A6=np.load(d+'act_L6.npy'); LG=np.load(d+'logits.npy'); PR=np.load(d+'probs.npy')
T=json.load(open('/work/test3_req.json')); pos=T['pos']
leak=0
for i,ps in enumerate(pos):
    bd=[{'.':0,'X':1,'O':2}[ch] for row in ps['board'] for ch in row]
    me=1 if ps['to_move']=='black' else 2
    for p in range(81):
        if bd[p]!=me and A5[i,324+p]>0: leak+=1
    order=np.argsort(-LG[i])
    for top in order:
        if top==81: print(i,'pass'); break
        nb2=play(bd,int(top),me)
        if nb2 is None: continue
        capt=sum(1 for q in range(81) if bd[q]==3-me and nb2[q]==0)
        g,l=group(nb2,int(top))
        print(i,divmod(int(top),9),'p=%.2f'%PR[i,top],'libs after',len(l),'grp',len(g),'captured',capt,'L6',[round(float(A6[i,j*81+top]),2) for j in range(6)])
        break
print('leak',leak)
```