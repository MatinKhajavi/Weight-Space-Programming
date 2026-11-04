
LIN=0;SAVE=81;SELF=162;ATT=243;EYE=324;INT=405
def offs(p,kind):
    r,c=divmod(p,9); out=[]
    ds={'a':[(-1,0),(1,0),(0,-1),(0,1)],'d':[(-1,-1),(-1,1),(1,-1),(1,1)],'j':[(-2,0),(2,0),(0,-2),(0,2)],
        'k':[(1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1)],'dd':[(-2,-2),(-2,2),(2,-2),(2,2)]}[kind]
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
    W,b=Z()
    for p in range(81):
        nbp=NB[p]; deg=len(nbp)
        def add_e(u,w):
            b[u]+=w*deg
            for q in nbp: W[OWN+q,u]-=w; W[OPP+q,u]-=w
        u=LIN+p; b[u]+=P['K']; add_e(u,P['w_e'])
        for q in nbp: W[U1+q,u]+=P['w_cap']; W[AXS+q,u]+=P['w_capS']
        for kind in ('a','d','j','k','dd'):
            for q in offs(p,kind):
                W[OWN+q,u]+=P['own_'+kind]; W[OPP+q,u]+=P['opp_'+kind]
        # own3 = OWN-LO ; own2 = LO-AO
        u=SAVE+p; b[u]-=9; add_e(u,1)
        for q in nbp: W[U1+q,u]+=8; W[OWN+q,u]+=2; W[LO+q,u]-=2
        u=SELF+p; b[u]+=2; add_e(u,-1)
        for q in nbp: W[OWN+q,u]-=2; W[LO+q,u]+=2; W[AXS+q,u]-=3
        u=ATT+p; b[u]-=5; add_e(u,1)
        for q in nbp: W[LX+q,u]+=4; W[U1+q,u]-=4; W[OWN+q,u]+=1; W[LO+q,u]-=1
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
P5=dict(w_att=2,K=12,w_e=0.2,w_cap=6,w_capS=0.4,
        own_a=-0.2,opp_a=0.5,own_d=0.3,opp_d=0.3,own_j=0.3,opp_j=0.2,own_k=0.2,opp_k=0.1,own_dd=0.0,opp_dd=0.0,
        w_save=8,w_self=-10,w_eye=-30,w_int=-3,int_T=2,int_own=1,int_N=0.05,int_M=10,
        prior=[-1.5,0,0.8,0.8,1.0],center_bonus=1.0,scale=1.5)
P5['pass']=-3
def build_all(P):
    Ws,bs=build_trunk()
    W6,b6,Wo,bo=build_head(P)
    Ws['L6']=W6; bs['L6']=b6; Ws['out']=Wo; bs['out']=bo
    return Ws,bs
def save_all(P):
    Ws,bs=build_all(P)
    for k in ['L1','L2','L3','L4','L5','L6','out']:
        save(k+'.W',Ws[k]); save(k+'.b',bs[k])
