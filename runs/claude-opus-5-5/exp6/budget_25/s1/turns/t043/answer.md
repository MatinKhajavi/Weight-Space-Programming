```python
import numpy as np, json
exec(open('/work/go.py').read())
d='/work/pass_7/'
A5=np.load(d+'act_L5.npy'); A6=np.load(d+'act_L6.npy'); LG=np.load(d+'logits.npy')
allp=json.load(open('/work/test2_req.json'))['positions']
st=dict(ao_fn=0,ao_fp=0,lo_fn=0,lo_fp=0,ax_fn=0,ax_fp=0,axs_fn=0,axs_fp=0,n_own=0,n_opp=0)
axvals=[]
for i,ps in enumerate(allp):
    bd=[{'.':0,'X':1,'O':2}[ch] for row in ps['board'] for ch in row]
    me=1 if ps['to_move']=='black' else 2
    for p in range(81):
        if bd[p]==0: continue
        g,l=group(bd,p)
        if bd[p]==me:
            st['n_own']+=1
            ao=A5[i,164+p]; lo=A5[i,245+p]
            if len(l)==1 and ao<.5: st['ao_fn']+=1
            if len(l)>1 and ao>.5: st['ao_fp']+=1
            if len(l)<=2 and lo<.5: st['lo_fn']+=1
            if len(l)>2 and lo>.5: st['lo_fp']+=1
        else:
            st['n_opp']+=1
            ax=A5[i,326+p]; axs=A5[i,407+p]
            if ax>0: axvals.append(ax)
            if len(l)==1 and ax<.5: st['ax_fn']+=1
            if len(l)>1 and ax>.5: st['ax_fp']+=1
            if len(l)==1 and len(g)<=2 and axs<.5: st['axs_fn']+=1
            if len(l)>1 and axs>.5: st['axs_fp']+=1
print(st)
print(np.round(sorted(axvals)[:5],3), np.round(sorted(axvals)[-5:],3))
# check L6 unit sanity on own carries: compare A5 own/opp carries with inputs
X=np.load(d+'inputs.npy')
print(np.abs(A5[:,0:81]-X[:,0::3]).max(), np.abs(A5[:,81:162]-X[:,1::3]).max())
```