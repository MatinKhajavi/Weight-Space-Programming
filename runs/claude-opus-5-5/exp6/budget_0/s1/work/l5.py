
def build_L5():
    Wm=np.zeros((H,H),np.float32); Bm=np.zeros(H,np.float32)
    def setu(tgt,ex):
        for k,v in ex.d.items(): Wm[k,tgt]+=v
        Bm[tgt]+=ex.c
    for p in pts:
        setu(U(0,p),V(U(0,p)))
        setu(U(1,p),V(U(1,p)))
        sig = 1 if par(p)==0 else 0
        for col in (0,1):
            ex=repD(col,sig,p)+V(U(col,p))-1
            for q in NB[p]:
                ex=ex+repD(col,sig,q)-V(U(col,q))
            setu(U(2+col,p),ex)
        deg=len(NB[p])
        ex=E(c=-deg)+3*V(U(1,p))            # lone opp stone, <=2 libs
        for q in NB[p]: ex=ex+V(U(0,q))-2*V(U(1,q))
        setu(U(4,p),ex)
        I=E()                                # deep-opponent-area ramp
        for q in pts:
            d=dist(p,q)
            if 1<=d<=3: I=I+{1:1.0,2:0.5,3:0.25}[d]*(V(U(0,q))-V(U(1,q)))
        setu(U(5,p),-1*I-2.5)
    return Wm,Bm
