```python
import json, numpy as np
exec(open('/work/goutil.py').read())
N=9
def onb(x): return 0<=x[0]<N and 0<=x[1]<N
pats=[]
for side in range(4):
    for t in range(1,8):
        if side==0: Bp=(0,t); e=(0,1); n_=(1,0)
        elif side==1: Bp=(8,t); e=(0,1); n_=(-1,0)
        elif side==2: Bp=(t,0); e=(1,0); n_=(0,1)
        else: Bp=(t,8); e=(1,0); n_=(0,-1)
        for s in (1,-1):
            Ap=(Bp[0]+s*e[0],Bp[1]+s*e[1]); Cp=(Bp[0]-s*e[0],Bp[1]-s*e[1])
            own=[Bp]+[X for X in ((Cp[0]-s*e[0],Cp[1]-s*e[1]),(Cp[0]+n_[0],Cp[1]+n_[1])) if onb(X)]
            opp=[X for X in ((Ap[0]+s*e[0],Ap[1]+s*e[1]),(Ap[0]+n_[0],Ap[1]+n_[1]),(Bp[0]+n_[0],Bp[1]+n_[1])) if onb(X)]
            pats.append((Ap,Cp,own,opp))
def match(b,col,A):
    for Ap,Cp,own,opp in pats:
        if Ap!=A: continue
        if b[Cp[0]*9+Cp[1]]!=0: continue
        if all(b[x[0]*9+x[1]]==col for x in own) and all(b[x[0]*9+x[1]]==-col for x in opp): return True
    return False
tot=0; hit=0
for dd in ('/work/pass_10/','/work/pass_14/','/work/pass_16/','/work/pass_12/'):
    G=json.load(open(dd+'games.json')); B=np.load(dd+'boards.npy')
    for g in G:
        if g['end']!='repetition' or g['result']!=-1: continue
        mv=g['moves']; n=len(mv); gi=g['game']; col=1 if g['net_color']=='black' else -1
        lastb=B[gi,n-1]
        rep=[k for k in range(n-2,max(-1,n-10),-1) if (B[gi,k]==lastb).all()]
        k=rep[0]; trig=mv[k+1]
        if trig['by']!='net' or trig['move']==81: continue
        tot+=1; m=match(B[gi,k],col,divmod(trig['move'],9)); hit+=m
        if not m: print(dd,gi,divmod(trig['move'],9))
print(hit,tot)
# also count how often the pattern fires on net's candidate moves generally (false alarms): count occurrences in boards where net to move
cnt=0; allp=0
for dd in ('/work/pass_14/',):
    G=json.load(open(dd+'games.json')); B=np.load(dd+'boards.npy')
    for g in G:
        gi=g['game']; col=1 if g['net_color']=='black' else -1
        for m in g['moves']:
            if m['by']!='net': continue
            k=m['ply']; b=B[gi,k-1] if k>0 else np.zeros(81,int)
            for Ap,Cp,own,opp in pats:
                if b[Ap[0]*9+Ap[1]]==0 and b[Cp[0]*9+Cp[1]]==0 and all(b[x[0]*9+x[1]]==col for x in own) and all(b[x[0]*9+x[1]]==-col for x in opp):
                    cnt+=1
                    nb,cap=play(b,Ap[0]*9+Ap[1],col)
                    if cap>0: allp+=1
print('pattern occurrences',cnt,'of which A captures',allp)
```