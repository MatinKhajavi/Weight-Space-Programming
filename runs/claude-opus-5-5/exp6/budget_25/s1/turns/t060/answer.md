```python
import numpy as np, json
exec(open('/work/go.py').read())
d='/work/pass_11/'
A5=np.load(d+'act_L5.npy'); A6=np.load(d+'act_L6.npy'); LG=np.load(d+'logits.npy')
T=json.load(open('/work/test3_req.json')); pos=T['pos']; meta=T['meta']
for i,(ps,(gi,k,mvp)) in enumerate(zip(pos,meta)):
    bd=[{'.':0,'X':1,'O':2}[ch] for row in ps['board'] for ch in row]
    me=1 if ps['to_move']=='black' else 2
    p=mvp
    l6=[round(float(A6[i,j*81+p]),2) for j in range(6)]
    info=[]
    for q in NB[p]:
        if bd[q]==0: continue
        gq,lq=group(bd,q)
        info.append((divmod(q,9),'own' if bd[q]==me else 'opp',len(gq),len(lq),
                     'U1=%.2f AXS=%.1f S3=%.1f LX=%.1f'%(A5[i,162+q],A5[i,243+q],A5[i,324+q],A5[i,405+q])))
    print(i,divmod(p,9),'logit %.1f pass %.1f'%(LG[i,p],LG[i,81]),'LIN,SAVE,SELF,ATT,EYE,INT',l6)
    for x in info: print('    ',x)
```