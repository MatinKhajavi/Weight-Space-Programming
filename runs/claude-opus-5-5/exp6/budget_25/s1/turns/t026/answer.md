```python
code=open('/work/trunk3.py').read()
code=code.replace("W[OPP+p,u]+=16384; b[u]=-16384","W[OPP+p,u]+=16384-1; b[u]=-16384+1")
# ensure OPP+p weight = 16384 exactly: it was +1 (mass) then += 16383 -> 16384 ; bias -16383 gives +1 mass
open('/work/trunk3.py','w').write(code)
head = r'''
LIN=0;SAVE=81;SELF=162;ATT=243;EYE=324;INT=405;C6=486
def offs(p,kind):
    r,c=divmod(p,9); out=[]
    if kind=='a': ds=[(-1,0),(1,0),(0,-1),(0,1)]
    elif kind=='d': ds=[(-1,-1),(-1,1),(1,-1),(1,1)]
    elif kind=='j': ds=[(-2,0),(2,0),(0,-2),(0,2)]
    elif kind=='k': ds=[(1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1)]
    elif kind=='dd': ds=[(-2,-2),(-2,2),(2,-2),(2,2)]
    for a,d in ds:
        if 0<=r+a<9 and 0<=c+d<9: out.append((r+a)*9+c+d)
    return out
def prior_of(P):
    pr=np.zeros(81,np.float32)
    for p in range(81):
        r,c=divmod(p,9); d=min(r,c,8-r,8-c); pr[p]=P['prior'][d]
    pr[40]+=P['center_bonus']
    return pr
def build_head(P):
    W,b=Z(); b[C6]=1
    for p in range(81):
        nbp=NB[p]; deg=len(nbp)
        def add_e(u,w):
            b[u]+=w*deg
            for q in nbp: W[OWN+q,u]-=w; W[OPP+q,u]-=w
        u=LIN+p; b[u]+=P['K']; add_e(u,P['w_e'])
        for q in nbp: W[AX+q,u]+=P['w_cap']; W[AXS+q,u]+=P['w_capS']
        for kind in ('a','d','j','k','dd'):
            for q in offs(p,kind):
                W[OWN+q,u]+=P['own_'+kind]; W[OPP+q,u]+=P['opp_'+kind]
        u=SAVE+p; b[u]-=5; add_e(u,1)
        for q in nbp: W[AO+q,u]+=4; W[OWN+q,u]+=1; W[AO+q,u]-=1
        u=SELF+p; b[u]+=2; add_e(u,-1)
        for q in nbp: W[OWN+q,u]-=1; W[AO+q,u]+=1; W[AXS+q,u]-=3
        u=ATT+p; b[u]-=5; add_e(u,1)
        for q in nbp: W[LX+q,u]+=4; W[AX+q,u]-=4; W[OWN+q,u]+=1; W[AO+q,u]-=1
        u=EYE+p; b[u]+=1-deg
        for q in nbp: W[OWN+q,u]+=1
        u=INT+p; b[u]-=P['int_T']
        for q in nbp: W[OWN+q,u]+=P['int_own']
        for q in range(81):
            W[OWN+q,u]+=P['int_N']; W[OPP+q,u]+=P['int_N']
            if md(p,q)<=2: W[OPP+q,u]-=P['int_M']
    Wo=np.zeros((512,82),np.float32); bo=np.zeros(82,np.float32)
    pr=prior_of(P)
    for p in range(81):
        Wo[LIN+p,p]=1; bo[p]=-P['K']+pr[p]
        Wo[SAVE+p,p]=P['w_save']; Wo[SELF+p,p]=P['w_self']; Wo[ATT+p,p]=P['w_att']
        Wo[EYE+p,p]=P['w_eye']; Wo[INT+p,p]=P['w_int']
    bo[81]=P['pass']
    s=P['scale']
    return W,b,Wo*s,bo*s
P3=dict(K=12,w_e=0.2,w_cap=8,w_capS=0.3,
        own_a=-0.2,opp_a=0.5,own_d=0.3,opp_d=0.3,own_j=0.3,opp_j=0.2,own_k=0.2,opp_k=0.1,own_dd=0.0,opp_dd=0.0,
        w_save=8,w_self=-6,w_att=2,w_eye=-30,w_int=-3,int_T=2,int_own=1,int_N=0.05,int_M=10,
        prior=[-1.5,0,0.8,0.8,1.0],center_bonus=1.0,scale=1.5)
P3['pass']=-3
def build_all(P):
    Ws,bs=build_trunk()
    W6,b6,Wo,bo=build_head(P)
    Ws['L6']=W6; bs['L6']=b6; Ws['out']=Wo; bs['out']=bo
    return Ws,bs
def save_all(P):
    Ws,bs=build_all(P)
    for k in ['L1','L2','L3','L4','L5','L6','out']:
        save(k+'.W',Ws[k]); save(k+'.b',bs[k])
'''
open('/work/head3.py','w').write(head)
g=dict(save=save); exec(code+head,g)
Ws,bs=g['build_all'](g['P3'])
print(Ws['L3'][81+5, 407+5], bs['L3'][407+5], {k:float(abs(Ws[k]).max()) for k in Ws})
g['save_all'](g['P3'])
open('/work/notes.txt','a').write("v3: trunk3.py+head3.py (exec both). channels SO,SX,TX,VX(mass for small-group atari AXS). dropped own lt3. P3 params. pass3 v2: strong 53% (17 reps), weak 97%\n")
```