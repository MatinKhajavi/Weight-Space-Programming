
PAR=dict(K=10.0,wcap=10,wsave=12,wn=0.7,wt=0.5,wemp=0.1,watk=1.5,wIX=-1.0,
         wSA=-13.0,wE2=-15.0,wIO=-2.0,wTE=-20.0,wZ0=-2.0,
         ring={0:-3.0,1:-1.5,2:0.3,3:0.8,4:2.0},passlogit=-4.0)
def DIAG(p):
    return [(p[0]+dr,p[1]+dc) for dr in (-1,1) for dc in (-1,1) if 0<=p[0]+dr<9 and 0<=p[1]+dc<9]
def build_L6_out(PA=PAR):
    W6=np.zeros((H,H),np.float32); B6=np.zeros(H,np.float32)
    def set6(tgt,ex):
        for k,v in ex.d.items(): W6[k,tgt]+=v
        B6[tgt]+=ex.c
    K=PA['K']
    for p in pts:
        nb=NB[p]; deg=len(nb); dg=DIAG(p)
        need = 3 if len(dg)==4 else len(dg)
        nO=Esum(V(U(0,q)) for q in nb); nX=Esum(V(U(1,q)) for q in nb)
        dO=Esum(V(U(0,q)) for q in dg)
        a=Esum(V(U(2,q)) for q in nb)
        cap=Esum(V(U(3,q)) for q in nb)
        atk=Esum(V(U(4,q)) for q in nb)
        n=deg-nO-nX; s=nO-a; t=nX-cap
        I=E(); emp=E(); nX2=E()
        for q in pts:
            d=dist(p,q)
            if 1<=d<=3:
                I=I+{1:1.0,2:0.5,3:0.25}[d]*(V(U(0,q))-V(U(1,q)))
            if 1<=d<=2:
                emp=emp+(1-V(U(0,q))-V(U(1,q)))
                nX2=nX2+V(U(1,q))
        set6(U(0,p), K+PA['wcap']*cap+PA['wsave']*a+PA['wn']*n+PA['wt']*t+PA['wemp']*emp
                      +PA['watk']*atk+PA['wIX']*V(U(5,p)))
        set6(U(1,p), 0.25+a-4*(n-1)-10*(s+cap))
        set6(U(2,p), 2-n-2*(t+cap+a))
        set6(U(3,p), I-0.75-3*nX-3*a)
        set6(U(4,p), 1+10*(nO-deg)+5*(dO-need)+a)      # true eye (never fill)
        set6(U(5,p), 1-n-2*cap-2*a)                  # n=0 fill without capture/save
    Wo=np.zeros((H,82),np.float32); Bo=np.zeros(82,np.float32)
    for p in pts:
        r,c=p; d=min(r,c,8-r,8-c)
        for t,w in ((0,1.0),(1,PA['wSA']),(2,PA['wE2']),(3,PA['wIO']),(4,PA['wTE']),(5,PA['wZ0'])):
            Wo[U(t,p),P(p)]=w
        Bo[P(p)]=-K+PA['ring'][d]
    Bo[81]=PA['passlogit']
    return W6,B6,Wo,Bo
