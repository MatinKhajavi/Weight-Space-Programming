```python
code5 = r'''
def build_L5():
    Wm=np.zeros((H,H),np.float32); Bm=np.zeros(H,np.float32)
    def setu(tgt,ex):
        for k,v in ex.d.items(): Wm[k,tgt]+=v
        Bm[tgt]+=ex.c
    for p in pts:
        setu(U(0,p),V(U(0,p)))
        setu(U(1,p),V(U(1,p)))
        sig = 1 if par(p)==0 else 0      # relevant "no-other-signal" bit for a stone next to an empty point
        for col in (0,1):
            ex=repD(col,sig,p)+V(U(col,p))-1
            for q in NB[p]:
                ex=ex+repD(col,sig,q)-V(U(col,q))
            setu(U(2+col,p),ex)          # 2: own atari-flag(rel), 3: opp atari-flag(rel)
        # lone opp stone with <=2 liberties: relu(-deg + nO - 2 nX + 3 X)
        deg=len(NB[p])
        ex=E(c=-deg)+3*V(U(1,p))
        for q in NB[p]:
            ex=ex+V(U(0,q))-2*V(U(1,q))
        setu(U(4,p),ex)
    return Wm,Bm
'''
open('/work/l5.py','w').write(code5)

code6 = r'''
PAR=dict(K=1.0,wcap=10,wsave=11,wn=0.7,wt=0.5,wemp=0.1,watk=1.5,
         wSA=-13.0,wE2=-15.0,wIO=-2.0,wIX=-1.0,wCS=0.5,
         ring={0:-2.5,1:-1.0,2:0.3,3:0.8,4:1.5},passlogit=-4.0)
def build_L6_out(PA=PAR):
    W6=np.zeros((H,H),np.float32); B6=np.zeros(H,np.float32)
    def set6(tgt,ex):
        for k,v in ex.d.items(): W6[k,tgt]+=v
        B6[tgt]+=ex.c
    K=PA['K']
    for p in pts:
        nb=NB[p]; deg=len(nb)
        nO=Esum(V(U(0,q)) for q in nb); nX=Esum(V(U(1,q)) for q in nb)
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
        set6(U(0,p), K+PA['wcap']*cap+PA['wsave']*a+PA['wn']*n+PA['wt']*t+PA['wemp']*emp+PA['watk']*atk)
        set6(U(1,p), 0.25+a-4*(n-1)-10*(s+cap))
        set6(U(2,p), 2-n-2*(t+cap+a))
        set6(U(3,p), I-0.75-3*nX-3*a)
        set6(U(4,p), -1*I-2.5)
        set6(U(5,p), nX2+12*cap-12)
    Wo=np.zeros((H,82),np.float32); Bo=np.zeros(82,np.float32)
    for p in pts:
        r,c=p; d=min(r,c,8-r,8-c)
        for t,w in ((0,1.0),(1,PA['wSA']),(2,PA['wE2']),(3,PA['wIO']),(4,PA['wIX']),(5,PA['wCS'])):
            Wo[U(t,p),P(p)]=w
        Bo[P(p)]=-K+PA['ring'][d]
    Bo[81]=PA['passlogit']
    return W6,B6,Wo,Bo
'''
open('/work/l6.py','w').write(code6)

exec(open('/work/base.py').read()); exec(open('/work/l5.py').read()); exec(open('/work/l6.py').read())
W1,B1=build_L1(inp)
W2,B2=prop(rep1); W3,B3=prop(repD); W4,B4=prop(repD)
W5,B5=build_L5()
W6,B6,Wo,Bo=build_L6_out()
def show_col(Wm,Bm,tgt,names):
    nz=np.nonzero(Wm[:,tgt])[0]
    items=[]
    for k in nz:
        t,pp=divmod(int(k),81)
        items.append(f"{names[t]}{divmod(pp,9)}:{Wm[k,tgt]:+g}")
    print(" ".join(items), "| bias", Bm[tgt])
nk=['O','X','WxO','WyO','WxX','WyX']; n5=['O','X','AO','AX','LX']
print("L5 AO (4,4) even:"); show_col(W5,B5,U(2,(4,4)),nk)
print("L5 AX (4,5) odd:"); show_col(W5,B5,U(3,(4,5)),nk)
print("L5 LX (0,0):"); show_col(W5,B5,U(4,(0,0)),nk)
print("L6 Lin (4,4):"); show_col(W6,B6,U(0,(4,4)),n5)
print("L6 SA (0,4):"); show_col(W6,B6,U(1,(0,4)),n5)
T={'L1.W':W1,'L1.b':B1,'L2.W':W2,'L2.b':B2,'L3.W':W3,'L3.b':B3,'L4.W':W4,'L4.b':B4,
   'L5.W':W5,'L5.b':B5,'L6.W':W6,'L6.b':B6,'out.W':Wo,'out.b':Bo}
for k,v in T.items():
    assert v.shape==tuple(SHAPES[k]); save(k,v.astype(np.float32))
print(saved())
```