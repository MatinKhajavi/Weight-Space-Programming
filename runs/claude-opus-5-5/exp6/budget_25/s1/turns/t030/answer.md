```python
import numpy as np, json
exec(open('/work/go.py').read())
d='/work/pass_4/'
A5=np.load(d+'act_L5.npy'); A6=np.load(d+'act_L6.npy'); LG=np.load(d+'logits.npy')
allp=json.load(open('/work/test2_req.json'))['positions']
st={'afn':0,'afp':0,'lfn':0,'lfp':0,'sfn':0,'sfp':0}
for i,ps in enumerate(allp):
    bd=[{'.':0,'X':1,'O':2}[ch] for row in ps['board'] for ch in row]
    me=1 if ps['to_move']=='black' else 2
    for p in range(81):
        if bd[p]==0: continue
        g,l=group(bd,p); own=bd[p]==me
        a=A5[i,(164 if own else 245)+p]
        if len(l)==1 and a<.5: st['afn']+=1
        if len(l)>1 and a>.5: st['afp']+=1
        if not own:
            lx=A5[i,326+p]; axs=A5[i,407+p]
            if len(l)<=2 and lx<.5: st['lfn']+=1
            if len(l)>2 and lx>.5: st['lfp']+=1
            if len(l)==1 and len(g)<=2 and axs<.5: st['sfn']+=1
            if len(l)>1 and axs>.5: st['sfp']+=1
print(st)
for i,mvp in ((30,(8,4)),(31,(2,0))):
    p=mvp[0]*9+mvp[1]
    print(i, 'logit',LG[i,p], 'LIN..INT', [A6[i,k*81+p] for k in range(6)])
    for q in NB[p]: print('  nbr',divmod(q,9), 'AO',A5[i,164+q],'AX',A5[i,245+q],'LX',A5[i,326+q],'AXS',A5[i,407+q])
```